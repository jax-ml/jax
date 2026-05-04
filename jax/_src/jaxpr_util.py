# Copyright 2020 The JAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utilities for the Jaxpr IR."""

from __future__ import annotations

import base64
from collections import Counter
from collections import defaultdict
from collections.abc import Callable, Iterator
import gzip
import html
import itertools
import json
import logging
import re
import types
from typing import Any, Union

from jax._src import config
from jax._src import core
from jax._src import path
from jax._src import source_info_util
from jax._src import util
from jax._src.lib import xla_client


map, unsafe_map = util.safe_map, map
zip, unsafe_zip = util.safe_zip, zip

logger = logging.getLogger(__name__)


def _all_eqns(
    jaxpr: core.Jaxpr, visited: set[core.Jaxpr] | None,
) -> Iterator[tuple[core.Jaxpr, core.JaxprEqn]]:
  for eqn in jaxpr.eqns:
    yield (jaxpr, eqn)
  for subjaxpr in core.subjaxprs(jaxpr):
    if visited is None:
      yield from _all_eqns(subjaxpr, visited)
    elif subjaxpr not in visited:
      visited.add(subjaxpr)
      yield from _all_eqns(subjaxpr, visited)

def all_eqns(
    jaxpr: core.Jaxpr, revisit_inner_jaxprs: bool = True
) -> Iterator[tuple[core.Jaxpr, core.JaxprEqn]]:
  yield from _all_eqns(jaxpr, None if revisit_inner_jaxprs else set())


def _all_eqns_with_traceback(
    jaxpr: core.Jaxpr, caller_tb: xla_client.Traceback | None,
    visited: set[core.Jaxpr]
) -> Iterator[tuple[xla_client.Traceback | None, core.JaxprEqn]]:
  for eqn in jaxpr.eqns:
    tb = eqn.source_info.traceback
    if caller_tb is not None:
      tb = caller_tb if tb is None else tb + caller_tb
    yield tb, eqn

    for subjaxpr in core.jaxprs_in_params(eqn.params):
      if subjaxpr not in visited:
        visited.add(subjaxpr)
        yield from _all_eqns_with_traceback(subjaxpr, tb, visited)


def collect_eqns(jaxpr: core.Jaxpr, key: Callable):
  d = defaultdict(list)
  for _, eqn in all_eqns(jaxpr):
    d[key(eqn)].append(eqn)
  return dict(d)

@util.weakref_lru_cache
def count_eqns(
    jaxpr: core.Jaxpr, revisit_inner_jaxprs: bool = True
) -> int:
  return sum(1 for _ in all_eqns(jaxpr, revisit_inner_jaxprs=revisit_inner_jaxprs))

def histogram(jaxpr: core.Jaxpr, key: Callable,
              key_fmt: Callable = lambda x: x):
  d = collect_eqns(jaxpr, key)
  return {key_fmt(k): len(v) for k, v in d.items()}

def primitives(jaxpr: core.Jaxpr):
  return histogram(jaxpr, lambda eqn: eqn.primitive.name)

def primitives_by_source(jaxpr: core.Jaxpr):
  def key(eqn):
    src = source_info_util.summarize(eqn.source_info)
    return (eqn.primitive.name, src)
  return histogram(jaxpr, key, ' @ '.join)

def primitives_by_shape(jaxpr: core.Jaxpr):
  def shape_fmt(var):
    return '*' if isinstance(var, core.DropVar) else var.aval.str_short()
  def key(eqn):
    return (eqn.primitive.name, ' '.join(map(shape_fmt, eqn.outvars)))
  return histogram(jaxpr, key, ' :: '.join)

def source_locations(jaxpr: core.Jaxpr):
  def key(eqn):
    return source_info_util.summarize(eqn.source_info)
  return histogram(jaxpr, key)

MaybeEqn = Union[core.JaxprEqn, None]

def var_defs_and_refs(jaxpr: core.Jaxpr):
  defs: dict[core.Var, MaybeEqn] = {}
  refs: dict[core.Var, list[MaybeEqn]] = {}

  def read(a: core.Atom, eqn: MaybeEqn):
    if not isinstance(a, core.Literal):
      assert a in defs, a
      assert a in refs, a
      refs[a].append(eqn)

  def write(v: core.Var, eqn: MaybeEqn):
    assert v not in defs, v
    assert v not in refs, v
    if not isinstance(v, core.DropVar):
      defs[v] = eqn
      refs[v] = []

  for v in jaxpr.constvars:
    write(v, None)
  for v in jaxpr.invars:
    write(v, None)

  for eqn in jaxpr.eqns:
    for a in eqn.invars:
      read(a, eqn)
    for v in eqn.outvars:
      write(v, eqn)

  for a in jaxpr.outvars:
    read(a, None)

  res = [(v, defs[v], refs[v]) for v in defs]
  subs = map(var_defs_and_refs, core.subjaxprs(jaxpr))
  return [(jaxpr, res), *subs] if subs else (jaxpr, res)


def _strip_workspace_root(filename: str, workspace_root: str) -> str:
  i = filename.rfind(workspace_root)
  return filename[i+len(workspace_root):] if i >= 0 else filename


def _pprof_profile(
    profile: dict[tuple[xla_client.Traceback | None, core.Primitive], int],
    workspace_root: str | None = None,
) -> bytes:
  """Converts a profile into a compressed pprof protocol buffer.

  The input profile is a map from (traceback, primitive) pairs to counts.
  """
  s: defaultdict[str, int]
  func: defaultdict[types.CodeType, int]
  loc: defaultdict[tuple[types.CodeType, int], int]

  s = defaultdict(itertools.count(1).__next__)
  func = defaultdict(itertools.count(1).__next__)
  loc = defaultdict(itertools.count(1).__next__)
  s[""] = 0
  primitive_key = s["primitive"]
  samples = []
  for (tb, primitive), count in profile.items():
    if tb is None:
      frames = []
    else:
      raw_frames = zip(*tb.raw_frames())
      frames = [loc[(code, lasti)] for code, lasti in raw_frames
                if source_info_util.is_user_filename(code.co_filename)]
    samples.append({
       "location_id": frames,
       "value": [count],
       "label": [{
         "key": primitive_key,
         "str": s[primitive.name]
        }]
    })

  locations = [
      {"id": loc_id,
       "line": [{"function_id": func[code],
                 "line": xla_client.Traceback.code_addr2line(code, lasti)}]}
      for (code, lasti), loc_id in loc.items()
  ]
  functions = []
  for code, func_id in func.items():
    filename = code.co_filename
    name = code.co_qualname
    if workspace_root is not None:
      filename = _strip_workspace_root(filename, workspace_root)
    else:
      pattern = config.hlo_source_file_canonicalization_regex.value
      if pattern:
        filename = re.sub(pattern, '', filename)
    name = f"{filename.removesuffix('.py').replace('/', '.')}.{name}"
    functions.append(
        {"id": func_id,
        "name": s[name],
        "filename": s[filename],
        "start_line": code.co_firstlineno}
    )
  sample_type = [{"type": s["equations"], "unit": s["count"]}]
  # This is the JSON encoding of a pprof profile protocol buffer. See:
  # https://github.com/google/pprof/blob/master/proto/profile.proto for a
  # description of the format.
  json_profile = json.dumps({
    "string_table": list(s.keys()),
    "location": locations,
    "function": functions,
    "sample_type": sample_type,
    "sample": samples,
  })
  return gzip.compress(xla_client._xla.json_to_pprof_profile(json_profile))


def pprof_equation_profile(jaxpr: core.Jaxpr, *,
                           workspace_root: str | None = None) -> bytes:
  """Generates a pprof profile that maps jaxpr equations to Python stack traces.

  By visualizing the profile using pprof, one can identify Python code that is
  responsible for yielding large numbers of jaxpr equations.

  Args:
    jaxpr: a Jaxpr.
    workspace_root: the root of the workspace. If specified, function names
      will be fully qualified, with respect to the workspace root.

  Returns:
    A gzip-compressed pprof Profile protocol buffer, suitable for passing to
    pprof tool for visualization.
  """
  d = Counter(
      (tb, eqn.primitive)
      for tb, eqn in _all_eqns_with_traceback(jaxpr, None, set())
  )
  return _pprof_profile(d, workspace_root)


def eqns_using_var_with_invar_index(jaxpr: core.Jaxpr, invar: core.Var) -> Iterator[tuple[core.JaxprEqn, int]]:
  """Find all the equations which use invar and the positional index of its binder"""
  for eqn in jaxpr.eqns:
    for invar_index, eqn_var in enumerate(eqn.invars):
      if eqn_var == invar:
        yield eqn, invar_index
        break # we found the var, no need to keep looking in this eqn

def jaxpr_and_binder_in_params(params, index: int) -> Iterator[tuple[core.Jaxpr, core.Var]]:
  for val in params.values():
    vals = val if isinstance(val, tuple) else (val,)
    for v in vals:
      if isinstance(v, core.Jaxpr):
        if index >= len(v.invars):
          raise RuntimeError(f"Failed to find index {index} in jaxpr.invars while building report")
        yield v, v.invars[index]
      elif isinstance(v, core.ClosedJaxpr):
        if index >= len(v.jaxpr.invars):
          raise RuntimeError(f"Failed to find index {index} in jaxpr.invars while building report")
        yield v.jaxpr, v.jaxpr.invars[index]

def eqns_using_var(jaxpr: core.Jaxpr, invar: core.Var) -> Iterator[core.JaxprEqn]:
  """Find the leaf equations using a variable"""
  # The complexity of this call is because the invar might originate from a nested jaxpr
  for eqn, invar_index in eqns_using_var_with_invar_index(jaxpr, invar):
    if (child_jaxprs_and_vars := tuple(jaxpr_and_binder_in_params(eqn.params, invar_index))):
      for (jaxpr, invar) in child_jaxprs_and_vars:
        yield from eqns_using_var(jaxpr, invar)
    else:
      # if the previous condition fails, there is no deeper jaxpr to explore =(
      yield eqn


_jaxpr_id_counter = itertools.count()

def maybe_dump_jaxpr_to_file(
    fun_name: str, jaxpr: core.Jaxpr
) -> str | None:
  """Maybe dumps the `jaxpr` to a file.

  Dumps the jaxpr if JAX_DUMP_JAXPR_TO is defined.

  Args:
    fn: The name of the function whose jaxpr is being dumped.
    jaxpr: The jaxpr to dump.

  Returns:
    The path to the file where the jaxpr was dumped, or None if no file was
    dumped.
  """
  if not (out_dir := path.make_jax_dump_dir(config.jax_dump_ir_to.value)):
    return None
  modes = config.jax_dump_ir_modes.value.split(",")
  if (
      "jaxpr" not in modes
      and "jaxpr_html" not in modes
      and "eqn_count_pprof" not in modes
  ):
    return None
  id = next(_jaxpr_id_counter)
  if "jaxpr" in modes:
    logging.log(
        logging.INFO, "Dumping jaxpr for %s to %s.", fun_name, out_dir
    )
    jaxpr_path = out_dir / f"jax_{id:06d}_{fun_name}.jaxpr.txt"
    jaxpr_path.write_text(jaxpr.pretty_print())
  if "jaxpr_html" in modes:
    logging.log(
        logging.INFO, "Dumping jaxpr HTML for %s to %s.", fun_name, out_dir
    )
    html_path = out_dir / f"jax_{id:06d}_{fun_name}.jaxpr.html"
    html_path.write_text(jaxpr_to_html(jaxpr))
  if "eqn_count_pprof" in modes:
    logging.log(
        logging.INFO, "Dumping eqn count pprof for %s to %s.", fun_name, out_dir
    )
    eqn_prof_path = out_dir / f"jax_{id:06d}_{fun_name}.eqn_count_pprof"
    eqn_prof_path.write_bytes(pprof_equation_profile(jaxpr))
  return fun_name


def jaxpr_to_html(jaxpr: core.Jaxpr) -> str:
  """Renders a Jaxpr as HTML with interactive tracebacks and search."""

  # 1. Render jaxpr to string and get source map
  source_map_output: list[list[tuple[int, int, Any]]] = []
  rendered_str = jaxpr.pretty_print(
      source_map=source_map_output, use_color=False
  )

  # 2. Process source map and build traceback DAG
  raw_frame_to_idx: dict[tuple[types.CodeType, int], int] = {}
  dag_nodes: list[dict[str, int | None]] = []
  node_to_idx: dict[tuple[int, int | None], int] = {}
  tb_to_node_idx: dict[Any, int | None] = {}

  def get_frame_idx(code: types.CodeType, lasti: int) -> int:
    key = (code, lasti)
    idx = raw_frame_to_idx.get(key)
    if idx is None:
      idx = len(raw_frame_to_idx)
      raw_frame_to_idx[key] = idx
    return idx

  def get_node_idx(frame_idx: int, parent_node_idx: int | None) -> int:
    key = (frame_idx, parent_node_idx)
    idx = node_to_idx.get(key)
    if idx is None:
      idx = len(dag_nodes)
      dag_nodes.append({"frame_idx": frame_idx, "parent": parent_node_idx})
      node_to_idx[key] = idx
    return idx

  def process_tb(tb: Any) -> int | None:
    idx = tb_to_node_idx.get(tb)
    if idx is not None:
      return idx

    code, lasti = tb.raw_frames()

    parent_node_idx = None
    # raw_frames gives inner to outer. We iterate from outer to inner.
    for i in reversed(range(len(code))):
      frame_idx = get_frame_idx(code[i], lasti[i])
      parent_node_idx = get_node_idx(frame_idx, parent_node_idx)

    tb_to_node_idx[tb] = parent_node_idx
    return parent_node_idx

  # 3. Generate HTML lines with spans
  lines = rendered_str.splitlines()
  html_lines = []

  for i, line in enumerate(lines):
    spans = source_map_output[i] if i < len(source_map_output) else []
    # Sort spans by start column
    spans.sort(key=lambda x: x[0])

    result = []
    last_idx = 0
    for start, end, tb in spans:
      if start > last_idx:
        result.append(html.escape(line[last_idx:start]))

      tb_node_idx = process_tb(tb)
      if tb_node_idx is not None:
        result.append(f'<span class="traceable" data-tb-idx="{tb_node_idx}">')
        result.append(html.escape(line[start:end]))
        result.append("</span>")
      else:
        result.append(html.escape(line[start:end]))
      last_idx = end

    if last_idx < len(line):
      result.append(html.escape(line[last_idx:]))

    html_lines.append("".join(result))

  # 4. Convert raw frames to final Frame representations with string pooling
  final_frames = []
  string_to_idx: dict[str, int] = {}

  def get_string_idx(s: str) -> int:
    idx = string_to_idx.get(s)
    if idx is None:
      idx = len(string_to_idx)
      string_to_idx[s] = idx
    return idx

  for code, lasti in raw_frame_to_idx:
    frame = source_info_util.raw_frame_to_frame(code, lasti)
    pattern = config.hlo_source_file_canonicalization_regex.value
    file_name = (
        re.sub(pattern, "", frame.file_name) if pattern else frame.file_name
    )
    final_frames.append({
        "file_idx": get_string_idx(file_name),
        "func_idx": get_string_idx(frame.function_name),
        "line": frame.start_line,
        "col": frame.start_column,
    })

  # 5. Construct final HTML and compress data

  data = {
      "frames": final_frames,
      "dag": dag_nodes,
      "strings": list(string_to_idx),
      "lines": html_lines,
  }

  json_data = json.dumps(data)
  compressed_data = gzip.compress(json_data.encode("utf-8"))
  base64_data = base64.b64encode(compressed_data).decode("utf-8")

  source_url_schema = config.source_url_schema.value or ""
  html_content = f"""
<!DOCTYPE html>
<html>
<head>
<style>
  body {{
    display: flex;
    font-family: monospace;
    margin: 0;
    height: 100vh;
  }}
  #jaxpr-container {{
    flex: 7;
    overflow: auto;
    padding: 10px;
    background-color: #f5f5f5;
    line-height: 18px;
    font-size: 14px;
  }}
  .line {{
    height: 18px;
    white-space: pre;
  }}
  #pane-divider {{
    width: 5px;
    cursor: col-resize;
    background-color: #ccc;
  }}
  #side-pane {{
    flex: 3;
    overflow: auto;
    padding: 10px;
    background-color: #fff;
    border-left: 1px solid #ccc;
  }}
  .traceable {{
    cursor: pointer;
    background-color: #e8f0fe;
  }}
  .traceable:hover {{
    background-color: #d2e3fc;
  }}
  .selected {{
    background-color: #aecbfa;
  }}
  .search-match {{
    background-color: #fff59d;
  }}
  .current-match {{
    background-color: #fff59d;
    border: 1px solid #f57f17;
    box-sizing: border-box;
  }}
  .frame {{
    margin-bottom: 8px;
    border-bottom: 1px solid #eee;
    padding-bottom: 4px;
  }}
  .frame-file {{ color: #5f6368; }}
  .frame-func {{ color: #1a73e8; font-weight: bold; }}
  .frame-loc {{ color: #80868b; }}
  #search-controls {{
    margin-bottom: 10px;
  }}
  #search-input {{
    width: 60%;
  }}
</style>
</head>
<body>

<div id="jaxpr-container">
  <div id="virtual-scroll-spacer" style="position: relative;">
    <div id="visible-lines-container" style="position: absolute; top: 0; left: 0; right: 0;"></div>
  </div>
</div>

<div id="pane-divider"></div>

<div id="side-pane">
  <h3>Search</h3>
  <div id="search-controls">
    <input type="text" id="search-input" placeholder="Search jaxpr...">
    <button id="search-prev">&lt;</button>
    <button id="search-next">&gt;</button>
    <span id="search-count"></span>
  </div>
  <div id="total-lines"></div>
  <hr>
  <h3>Traceback</h3>
  <div id="traceback-content">Click on a shaded line to see the traceback.</div>
</div>

<script>
  async function decompress(base64Data) {{
    const binary = atob(base64Data);
    const bytes = new Uint8Array(binary.length);
    for (let i = 0; i < binary.length; i++) {{
      bytes[i] = binary.charCodeAt(i);
    }}
    const stream = new Response(bytes).body.pipeThrough(new DecompressionStream('gzip'));
    const result = await new Response(stream).text();
    return JSON.parse(result);
  }}

  const base64Data = "{base64_data}";
  const sourceUrlSchema = "{source_url_schema}";

  decompress(base64Data).then(data => {{
    const frames = data.frames;
    const dag = data.dag;
    const strings = data.strings;
    const allLines = data.lines;

    document.getElementById('total-lines').textContent = `Total lines: ${{allLines.length}}`;

    const lineHeight = 18;
    const container = document.getElementById('jaxpr-container');
    const spacer = document.getElementById('virtual-scroll-spacer');
    const visibleContainer = document.getElementById('visible-lines-container');

    const MAX_HEIGHT = 10000000; // Reduce to 10M pixels to be safer with browser limits
    const totalHeight = allLines.length * lineHeight;
    const useScaling = totalHeight > MAX_HEIGHT;
    const spacerHeight = useScaling ? MAX_HEIGHT : totalHeight;

    spacer.style.height = spacerHeight + 'px';

    const buffer = 5;
    let matchingLines = [];
    let currentMatchIdx = -1;

    function renderVisibleLines() {{
      const scrollTop = container.scrollTop;
      const containerHeight = container.clientHeight;

      const scale = useScaling ? ((totalHeight - containerHeight) / (MAX_HEIGHT - containerHeight)) : 1.0;
      const virtualScrollTop = scrollTop * scale;

      const startIdx = Math.floor(virtualScrollTop / lineHeight);
      const offset = virtualScrollTop % lineHeight;

      const renderedStartIdx = Math.max(0, startIdx - buffer);
      const actualBuffer = startIdx - renderedStartIdx;

      const endIdx = Math.min(allLines.length, Math.ceil((virtualScrollTop + containerHeight) / lineHeight) + buffer);

      visibleContainer.innerHTML = allLines.slice(renderedStartIdx, endIdx).map((line, idx) => {{
        const lineAbsoluteIdx = renderedStartIdx + idx;
        const isMatch = matchingLines.includes(lineAbsoluteIdx);
        const isCurrent = currentMatchIdx !== -1 && lineAbsoluteIdx === matchingLines[currentMatchIdx];

        let className = "line";
        if (isMatch) className += " search-match";
        if (isCurrent) className += " current-match";

        return `<div class="${{className}}">${{line}}</div>`;
      }}).join('');

      if (useScaling) {{
        visibleContainer.style.top = (scrollTop - offset - (actualBuffer * lineHeight)) + 'px';
      }} else {{
        visibleContainer.style.top = (renderedStartIdx * lineHeight) + 'px';
      }}
    }}

    container.addEventListener('scroll', renderVisibleLines);
    window.addEventListener('resize', renderVisibleLines);
    renderVisibleLines();

    // Event Delegation
    let selectedElement = null;
    container.addEventListener('click', (e) => {{
      const traceable = e.target.closest('.traceable');
      if (traceable) {{
        if (selectedElement) {{
          selectedElement.classList.remove('selected');
        }}
        traceable.classList.add('selected');
        selectedElement = traceable;
        const tbIdx = parseInt(traceable.getAttribute('data-tb-idx'));
        renderTraceback(tbIdx);
      }}
    }});

    function renderTraceback(nodeIdx) {{
      const contentDiv = document.getElementById('traceback-content');
      contentDiv.innerHTML = '';

      let currentIdx = nodeIdx;
      const renderedFrames = [];

      while (currentIdx !== null && currentIdx !== undefined) {{
        const node = dag[currentIdx];
        const frame = frames[node.frame_idx];
        const file = strings[frame.file_idx];
        const func = strings[frame.func_idx];
        renderedFrames.push({{file: file, func: func, line: frame.line, col: frame.col}});
        currentIdx = node.parent;
      }}

      renderedFrames.reverse();

      renderedFrames.forEach(frame => {{
        const frameDiv = document.createElement('div');
        frameDiv.className = 'frame';
        if (sourceUrlSchema) {{
          const url = sourceUrlSchema.replace('{{file}}', frame.file).replace('{{line}}', frame.line);
          frameDiv.innerHTML = `
            <a href="${{url}}" target="_blank">${{escapeHtml(frame.file)}}:${{frame.line}}</a>
            in <span class="frame-func">${{escapeHtml(frame.func)}}</span>
          `;
        }} else {{
          frameDiv.innerHTML = `
            <span class="frame-file">${{escapeHtml(frame.file)}}:${{frame.line}}</span>
            in <span class="frame-func">${{escapeHtml(frame.func)}}</span>
          `;
        }}
        contentDiv.appendChild(frameDiv);
      }});

      if (renderedFrames.length === 0) {{
        contentDiv.innerHTML = 'No traceback information available.';
      }}
    }}

    function escapeHtml(text) {{
      const div = document.createElement('div');
      div.textContent = text;
      return div.innerHTML;
    }}

    // Search Logic
    const searchInput = document.getElementById('search-input');
    const searchPrev = document.getElementById('search-prev');
    const searchNext = document.getElementById('search-next');
    const searchCount = document.getElementById('search-count');

    function performSearch() {{
      const query = searchInput.value.trim().toLowerCase();
      matchingLines = [];
      currentMatchIdx = -1;

      if (query) {{
        allLines.forEach((line, idx) => {{
          // Strip HTML tags for searching
          const text = line.replace(/<[^>]*>/g, '').toLowerCase();
          if (text.includes(query)) {{
            matchingLines.push(idx);
          }}
        }});
      }}

      updateSearchUI();
      renderVisibleLines();
    }}

    function updateSearchUI() {{
      if (matchingLines.length > 0) {{
        if (currentMatchIdx === -1) currentMatchIdx = 0;
        searchCount.textContent = `${{currentMatchIdx + 1}} / ${{matchingLines.length}}`;
      }} else {{
        searchCount.textContent = searchInput.value.trim() ? "0 / 0" : "";
        currentMatchIdx = -1;
      }}
    }}

    function goToMatch(idx) {{
      if (matchingLines.length === 0) return;
      currentMatchIdx = (idx + matchingLines.length) % matchingLines.length;
      updateSearchUI();

      const lineIdx = matchingLines[currentMatchIdx];
      const scale = useScaling ? (totalHeight / MAX_HEIGHT) : 1.0;
      const virtualScrollTop = lineIdx * lineHeight;
      const scrollTop = useScaling ? (virtualScrollTop / scale) : virtualScrollTop;

      container.scrollTop = scrollTop;
      renderVisibleLines();
    }}

    searchInput.addEventListener('input', performSearch);
    searchPrev.addEventListener('click', () => goToMatch(currentMatchIdx - 1));
    searchNext.addEventListener('click', () => goToMatch(currentMatchIdx + 1));
  }});

  // Simple resizable pane logic
  const divider = document.getElementById('pane-divider');
  const leftPane = document.getElementById('jaxpr-container');
  const rightPane = document.getElementById('side-pane');

  let isResizing = false;

  divider.addEventListener('mousedown', (e) => {{
    isResizing = true;
    document.body.style.cursor = 'col-resize';
    e.preventDefault();
  }});

  document.addEventListener('mousemove', (e) => {{
    if (!isResizing) return;
    const offsetRight = document.body.clientWidth - e.clientX;

    if (offsetRight > 100 && offsetRight < document.body.clientWidth - 100) {{
      rightPane.style.flex = 'none';
      rightPane.style.width = offsetRight + 'px';
    }}
  }});

  document.addEventListener('mouseup', () => {{
    isResizing = false;
    document.body.style.cursor = 'default';
  }});
</script>

</body>
</html>
"""
  return html_content
