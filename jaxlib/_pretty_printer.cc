/* Copyright 2025 The JAX Authors

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <algorithm>
#include <cstddef>
#include <new>
#include <optional>
#include <stack>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/inlined_vector.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "nanobind/nanobind.h"
#include "nanobind/stl/optional.h"  // IWYU pragma: keep
#include "nanobind/stl/string.h"  // IWYU pragma: keep
#include "nanobind/stl/vector.h"  // IWYU pragma: keep

namespace nb = nanobind;

namespace jax {
namespace {

enum class Color {
  kBlack = 30,
  kRed = 31,
  kGreen = 32,
  kYellow = 33,
  kBlue = 34,
  kMagenta = 35,
  kCyan = 36,
  kWhite = 37,
  kReset = 39,
};

std::string ColorToString(Color color) {
  switch (color) {
    case Color::kBlack:
      return "black";
    case Color::kRed:
      return "red";
    case Color::kGreen:
      return "green";
    case Color::kYellow:
      return "yellow";
    case Color::kBlue:
      return "blue";
    case Color::kMagenta:
      return "magenta";
    case Color::kCyan:
      return "cyan";
    case Color::kWhite:
      return "white";
    case Color::kReset:
      return "reset";
  }
}

enum class Intensity {
  kNormal = 22,
  kDim = 2,
  kBright = 1,
};

std::string IntensityToString(Intensity intensity) {
  switch (intensity) {
    case Intensity::kNormal:
      return "normal";
    case Intensity::kDim:
      return "dim";
    case Intensity::kBright:
      return "bright";
  }
}

enum class OutputFormat {
  kText,
  kHtml,
};

struct FormatState;
struct FormatAgendum;

class Doc {
 public:
  Doc(int num_annotations) : num_annotations_(num_annotations) {}
  virtual ~Doc() = default;
  virtual std::string Repr() const = 0;

  int num_annotations() const { return num_annotations_; }

  virtual void Fits(std::stack<const Doc*>& agenda, int& width) const = 0;

  // Returns true if the doc may be sparse, i.e. there are no breaks between
  // annotations. Returns false if the doc is known not to be sparse.
  virtual bool Sparse(std::stack<const Doc*>& agenda, int& num_annotations,
                      bool& seen_break) const = 0;

  virtual void Format(const FormatAgendum& agendum,
                      FormatState& state) const = 0;

  static PyTypeObject* type;
  static PyType_Slot tp_slots[];

 private:
  int num_annotations_;
};

struct DocObject {
  PyObject_HEAD;
};

Doc* GetDoc(nb::handle h) {
  return std::launder(reinterpret_cast<Doc*>(reinterpret_cast<char*>(h.ptr()) +
                                             sizeof(DocObject)));
}

class PyDoc : public nb::object {
 public:
  NB_OBJECT(PyDoc, nb::object, "Doc", Check);

  static bool Check(nb::handle h) {
    return Doc::type && PyObject_TypeCheck(h.ptr(), Doc::type);
  }

  Doc* operator->() const { return GetDoc(*this); }
  Doc& operator*() const { return *GetDoc(*this); }
  Doc* get() const { return ptr() ? GetDoc(*this) : nullptr; }
};

class NilDoc final : public Doc {
 public:
  NilDoc() : Doc(/*num_annotations=*/0) {}
  std::string Repr() const override;

  void Fits(std::stack<const Doc*>& agenda, int& width) const override;
  bool Sparse(std::stack<const Doc*>& agenda, int& num_annotations,
              bool& seen_break) const override;
  virtual void Format(const FormatAgendum& agendum,
                      FormatState& state) const override;
  static PyTypeObject* type;
  static PyType_Slot tp_slots[];
};

class TextDoc final : public Doc {
 public:
  TextDoc(std::string text, std::optional<std::string> annotation)
      : Doc(annotation.has_value() ? 1 : 0),
        text_(std::move(text)),
        annotation_(std::move(annotation)) {}
  std::string Repr() const override;
  void Fits(std::stack<const Doc*>& agenda, int& width) const override;
  bool Sparse(std::stack<const Doc*>& agenda, int& num_annotations,
              bool& seen_break) const override;
  virtual void Format(const FormatAgendum& agendum,
                      FormatState& state) const override;
  static PyTypeObject* type;
  static PyType_Slot tp_slots[];

 private:
  std::string text_;
  std::optional<std::string> annotation_;
};

class ConcatDoc final : public Doc {
 public:
  explicit ConcatDoc(std::vector<PyDoc> children)
      : Doc(TotalNumAnnotations(children)), children_(std::move(children)) {}
  std::string Repr() const override;

  void Fits(std::stack<const Doc*>& agenda, int& width) const override;
  bool Sparse(std::stack<const Doc*>& agenda, int& num_annotations,
              bool& seen_break) const override;
  virtual void Format(const FormatAgendum& agendum,
                      FormatState& state) const override;
  static PyTypeObject* type;
  static PyType_Slot tp_slots[];

 private:
  static int TotalNumAnnotations(absl::Span<const PyDoc> children) {
    int total = 0;
    for (const auto& child : children) {
      total += child->num_annotations();
    }
    return total;
  }
  std::vector<PyDoc> children_;
};

class BreakDoc final : public Doc {
 public:
  explicit BreakDoc(std::string text)
      : Doc(/*num_annotations=*/0), text_(std::move(text)) {}
  std::string Repr() const override;
  void Fits(std::stack<const Doc*>& agenda, int& width) const override;
  bool Sparse(std::stack<const Doc*>& agenda, int& num_annotations,
              bool& seen_break) const override;
  virtual void Format(const FormatAgendum& agendum,
                      FormatState& state) const override;
  static PyTypeObject* type;
  static PyType_Slot tp_slots[];

 private:
  std::string text_;
};

class GroupDoc final : public Doc {
 public:
  explicit GroupDoc(PyDoc child)
      : Doc(/*num_annotations=*/child->num_annotations()),
        child_(std::move(child)) {}
  std::string Repr() const override;
  void Fits(std::stack<const Doc*>& agenda, int& width) const override;
  bool Sparse(std::stack<const Doc*>& agenda, int& num_annotations,
              bool& seen_break) const override;
  virtual void Format(const FormatAgendum& agendum,
                      FormatState& state) const override;
  static PyTypeObject* type;
  static PyType_Slot tp_slots[];

 private:
  PyDoc child_;
};

class NestDoc final : public Doc {
 public:
  explicit NestDoc(int n, PyDoc child)
      : Doc(child->num_annotations()), n_(n), child_(std::move(child)) {}
  std::string Repr() const override;
  void Fits(std::stack<const Doc*>& agenda, int& width) const override;
  bool Sparse(std::stack<const Doc*>& agenda, int& num_annotations,
              bool& seen_break) const override;
  virtual void Format(const FormatAgendum& agendum,
                      FormatState& state) const override;
  static PyTypeObject* type;
  static PyType_Slot tp_slots[];

 private:
  int n_;
  PyDoc child_;
};

class SourceMapDoc final : public Doc {
 public:
  explicit SourceMapDoc(PyDoc child, nb::object source)
      : Doc(child->num_annotations()),
        child_(std::move(child)),
        source_(std::move(source)) {}
  std::string Repr() const override;
  void Fits(std::stack<const Doc*>& agenda, int& width) const override;
  bool Sparse(std::stack<const Doc*>& agenda, int& num_annotations,
              bool& seen_break) const override;
  virtual void Format(const FormatAgendum& agendum,
                      FormatState& state) const override;
  static PyTypeObject* type;
  static PyType_Slot tp_slots[];

 private:
  PyDoc child_;
  nb::object source_;
};

class ColorDoc final : public Doc {
 public:
  explicit ColorDoc(PyDoc child, std::optional<Color> foreground,
                    std::optional<Color> background,
                    std::optional<Intensity> intensity)
      : Doc(child->num_annotations()),
        child_(std::move(child)),
        foreground_(foreground),
        background_(background),
        intensity_(intensity) {}

  std::string Repr() const override;
  void Fits(std::stack<const Doc*>& agenda, int& width) const override;
  bool Sparse(std::stack<const Doc*>& agenda, int& num_annotations,
              bool& seen_break) const override;
  virtual void Format(const FormatAgendum& agendum,
                      FormatState& state) const override;
  static PyTypeObject* type;
  static PyType_Slot tp_slots[];

 private:
  PyDoc child_;
  std::optional<Color> foreground_;
  std::optional<Color> background_;
  std::optional<Intensity> intensity_;
};

template <typename T>
struct DocInstance {
  DocObject base;
  T doc;

  static_assert(sizeof(DocObject) % alignof(T) == 0);
};

template <typename T>
void Doc_tp_dealloc(PyObject* self) {
  reinterpret_cast<DocInstance<T>*>(self)->doc.~T();
  Py_TYPE(self)->tp_free(self);
}

template <typename T, typename... Args>
PyDoc MakeDoc(Args&&... args) {
  DocInstance<T>* self = PyObject_New(DocInstance<T>, T::type);
  new (&self->doc) T(std::forward<Args>(args)...);
  return nb::steal<PyDoc>(reinterpret_cast<PyObject*>(self));
}

PyTypeObject* Doc::type = nullptr;
PyTypeObject* NilDoc::type = nullptr;
PyTypeObject* TextDoc::type = nullptr;
PyTypeObject* ConcatDoc::type = nullptr;
PyTypeObject* BreakDoc::type = nullptr;
PyTypeObject* GroupDoc::type = nullptr;
PyTypeObject* NestDoc::type = nullptr;
PyTypeObject* SourceMapDoc::type = nullptr;
PyTypeObject* ColorDoc::type = nullptr;

PyObject* Doc_tp_new(PyTypeObject* type, PyObject* args, PyObject* kwds) {
  PyErr_SetString(PyExc_TypeError,
                  "Doc is abstract and cannot be instantiated");
  return nullptr;
}

PyType_Slot Doc::tp_slots[] = {
    {Py_tp_new, reinterpret_cast<void*>(Doc_tp_new)},
    {0, nullptr},
};

PyType_Slot NilDoc::tp_slots[] = {
    {Py_tp_dealloc, reinterpret_cast<void*>(&Doc_tp_dealloc<NilDoc>)},
    {0, nullptr},
};

PyType_Slot TextDoc::tp_slots[] = {
    {Py_tp_dealloc, reinterpret_cast<void*>(&Doc_tp_dealloc<TextDoc>)},
    {0, nullptr},
};

PyType_Slot ConcatDoc::tp_slots[] = {
    {Py_tp_dealloc, reinterpret_cast<void*>(&Doc_tp_dealloc<ConcatDoc>)},
    {0, nullptr},
};

PyType_Slot BreakDoc::tp_slots[] = {
    {Py_tp_dealloc, reinterpret_cast<void*>(&Doc_tp_dealloc<BreakDoc>)},
    {0, nullptr},
};

PyType_Slot GroupDoc::tp_slots[] = {
    {Py_tp_dealloc, reinterpret_cast<void*>(&Doc_tp_dealloc<GroupDoc>)},
    {0, nullptr},
};

PyType_Slot NestDoc::tp_slots[] = {
    {Py_tp_dealloc, reinterpret_cast<void*>(&Doc_tp_dealloc<NestDoc>)},
    {0, nullptr},
};

PyType_Slot SourceMapDoc::tp_slots[] = {
    {Py_tp_dealloc, reinterpret_cast<void*>(&Doc_tp_dealloc<SourceMapDoc>)},
    {0, nullptr},
};

PyType_Slot ColorDoc::tp_slots[] = {
    {Py_tp_dealloc, reinterpret_cast<void*>(&Doc_tp_dealloc<ColorDoc>)},
    {0, nullptr},
};

std::string NilDoc::Repr() const { return "nil"; }

std::string TextDoc::Repr() const {
  if (annotation_.has_value()) {
    return absl::StrFormat("text(\"%s\", annotation=\"%s\")", text_,
                           *annotation_);
  } else {
    return absl::StrFormat("text(\"%s\")", text_);
  }
}

std::string ConcatDoc::Repr() const {
  return absl::StrFormat(
      "concat(%s)",
      absl::StrJoin(children_, ", ", [](std::string* out, const auto& child) {
        absl::StrAppend(out, child->Repr());
      }));
}

std::string BreakDoc::Repr() const {
  return absl::StrFormat("break(\"%s\")", text_);
}

std::string GroupDoc::Repr() const {
  return absl::StrFormat("group(%s)", child_->Repr());
}

std::string NestDoc::Repr() const {
  return absl::StrFormat("nest(%d, %s)", n_, child_->Repr());
}

std::string SourceMapDoc::Repr() const {
  return absl::StrFormat("source(%s, %s)", child_->Repr(),
                         nb::cast<std::string>(nb::repr(source_)));
}

std::string ColorDoc::Repr() const {
  std::string foreground_str =
      foreground_.has_value() ? ColorToString(*foreground_) : "None";
  std::string background_str =
      background_.has_value() ? ColorToString(*background_) : "None";
  std::string intensity_str =
      intensity_.has_value() ? IntensityToString(*intensity_) : "None";
  return absl::StrFormat("color(%s, %s, %s, %s)", child_->Repr(),
                         foreground_str, background_str, intensity_str);
}

// Fits method implementations

void NilDoc::Fits(std::stack<const Doc*>& agenda, int& width) const {}

void TextDoc::Fits(std::stack<const Doc*>& agenda, int& width) const {
  width -= text_.size();
}

void ConcatDoc::Fits(std::stack<const Doc*>& agenda, int& width) const {
  for (auto it = children_.rbegin(); it != children_.rend(); ++it) {
    agenda.push(it->get());
  }
}

void BreakDoc::Fits(std::stack<const Doc*>& agenda, int& width) const {
  width -= static_cast<int>(text_.size());
}

void GroupDoc::Fits(std::stack<const Doc*>& agenda, int& width) const {
  agenda.push(child_.get());
}

void NestDoc::Fits(std::stack<const Doc*>& agenda, int& width) const {
  agenda.push(child_.get());
}

void SourceMapDoc::Fits(std::stack<const Doc*>& agenda, int& width) const {
  agenda.push(child_.get());
}

void ColorDoc::Fits(std::stack<const Doc*>& agenda, int& width) const {
  agenda.push(child_.get());
}

bool Fits(const Doc* doc, int width) {
  std::stack<const Doc*> agenda;
  agenda.push(doc);
  while (width >= 0 && !agenda.empty()) {
    const Doc* doc = agenda.top();
    agenda.pop();
    doc->Fits(agenda, width);
  }
  return width >= 0;
}

// Sparse method implementations

bool NilDoc::Sparse(std::stack<const Doc*>& agenda, int& num_annotations,
                    bool& seen_break) const {
  return true;
}

bool TextDoc::Sparse(std::stack<const Doc*>& agenda, int& num_annotations,
                     bool& seen_break) const {
  if (annotation_.has_value()) {
    if (num_annotations >= 1 && seen_break) {
      return false;
    }
    num_annotations -= 1;
  }
  return true;
}

bool ConcatDoc::Sparse(std::stack<const Doc*>& agenda, int& num_annotations,
                       bool& seen_break) const {
  for (auto it = children_.rbegin(); it != children_.rend(); ++it) {
    agenda.push(it->get());
  }
  return true;
}

bool BreakDoc::Sparse(std::stack<const Doc*>& agenda, int& num_annotations,
                      bool& seen_break) const {
  seen_break = true;
  return true;
}

bool GroupDoc::Sparse(std::stack<const Doc*>& agenda, int& num_annotations,
                      bool& seen_break) const {
  agenda.push(child_.get());
  return true;
}

bool NestDoc::Sparse(std::stack<const Doc*>& agenda, int& num_annotations,
                     bool& seen_break) const {
  agenda.push(child_.get());
  return true;
}

bool SourceMapDoc::Sparse(std::stack<const Doc*>& agenda, int& num_annotations,
                          bool& seen_break) const {
  agenda.push(child_.get());
  return true;
}

bool ColorDoc::Sparse(std::stack<const Doc*>& agenda, int& num_annotations,
                      bool& seen_break) const {
  agenda.push(child_.get());
  return true;
}

// Returns true if the doc is sparse, i.e. there are no breaks between
// annotations.
bool Sparse(const Doc* doc) {
  if (doc->num_annotations() == 0) {
    return true;
  }
  std::stack<const Doc*> agenda;
  agenda.push(doc);
  int num_annotations = 0;
  bool seen_break = false;
  while (!agenda.empty()) {
    const Doc* doc = agenda.top();
    agenda.pop();
    if (!doc->Sparse(agenda, num_annotations, seen_break)) {
      return false;
    }
  }
  return true;
}

struct ColorState {
  Color foreground;
  Color background;
  Intensity intensity;

  bool operator==(const ColorState& other) const {
    return foreground == other.foreground && background == other.background &&
           intensity == other.intensity;
  }
  bool operator!=(const ColorState& other) const { return !operator==(other); }
};

constexpr ColorState kDefaultColors =
    ColorState{Color::kReset, Color::kReset, Intensity::kNormal};
constexpr ColorState kAnnotationColors =
    ColorState{Color::kReset, Color::kReset, Intensity::kDim};

enum class BreakMode { kFlat, kBreak };

struct FormatAgendum {
  int indent;
  BreakMode mode;
  const Doc* doc;
  ColorState color;
  nb::object source;
};

struct Line {
  std::string text;
  int width;
  std::vector<std::string> annotations;
};

// Format method implementations

struct FormatState {
  int width;
  std::stack<FormatAgendum> agenda;
  std::string line_text;
  int k;
  std::vector<std::string> line_annotations;
  std::optional<ColorState> color;
  std::optional<nb::list> source_map;
  nb::list line_source_map;
  int source_start;
  nb::object source;
  std::vector<Line> lines;
  OutputFormat output_format = OutputFormat::kText;

  // If true, color/style is reset to default at the end of each line and
  // restored at the start of the next line. This makes each line in the output
  // cleanly splittable (independent) of other lines.
  bool separable_lines = false;
};

void EscapeHtml(std::string* out, absl::string_view data) {
  out->reserve(out->size() + data.size());
  for (size_t pos = 0; pos != data.size(); ++pos) {
    switch (data[pos]) {
      case '&':
        out->append("&amp;");
        break;
      case '\"':
        out->append("&quot;");
        break;
      case '\'':
        out->append("&apos;");
        break;
      case '<':
        out->append("&lt;");
        break;
      case '>':
        out->append("&gt;");
        break;
      default:
        out->append(&data[pos], 1);
        break;
    }
  }
}

std::string GetHtmlSpanOpeningTag(const ColorState& state) {
  if (state == kDefaultColors) return "";
  std::string result = "<span class=\"";
  absl::InlinedVector<std::string, 3> classes;
  if (state.foreground != Color::kReset) {
    classes.push_back(
        absl::StrCat("ansi-fg-", static_cast<int>(state.foreground)));
  }
  if (state.background != Color::kReset) {
    classes.push_back(
        absl::StrCat("ansi-bg-", static_cast<int>(state.background) + 10));
  }
  if (state.intensity != Intensity::kNormal) {
    classes.push_back(
        absl::StrCat("ansi-intensity-", static_cast<int>(state.intensity)));
  }
  absl::StrAppend(&result, absl::StrJoin(classes, " "), "\">");
  return result;
}

std::string UpdateColorAnsi(std::optional<ColorState>& state,
                            const ColorState& update) {
  if (!state.has_value() || *state == update) {
    return "";
  }
  std::string result = "\033[";
  absl::InlinedVector<std::string, 3> codes;
  if (state->foreground != update.foreground) {
    codes.push_back(absl::StrCat(static_cast<int>(update.foreground)));
  }
  if (state->background != update.background) {
    codes.push_back(absl::StrCat(static_cast<int>(update.background) + 10));
  }
  if (state->intensity != update.intensity) {
    codes.push_back(absl::StrCat(static_cast<int>(update.intensity)));
  }
  absl::StrAppend(&result, absl::StrJoin(codes, ";"), "m");
  state = update;
  return result;
}

std::string UpdateColorHtml(std::optional<ColorState>& state,
                            const ColorState& update) {
  if (!state.has_value() || *state == update) {
    return "";
  }
  std::string result;
  if (*state != kDefaultColors) {
    result.append("</span>");
  }
  if (update != kDefaultColors) {
    result.append(GetHtmlSpanOpeningTag(update));
  }
  state = update;
  return result;
}

std::string UpdateColor(std::optional<ColorState>& color, OutputFormat format,
                        const ColorState& update) {
  if (format == OutputFormat::kHtml) {
    return UpdateColorHtml(color, update);
  } else {
    return UpdateColorAnsi(color, update);
  }
}

void EndLine(FormatState& state, int next_indent) {
  if (state.source_map.has_value()) {
    // We want to ensure that source map boundaries do not fall in the middle
    // of color regions so if we, e.g., wrap them in an HTML tag, then the tags
    // are properly nested. Source map boundaries always end at line boundaries
    // so we reset to the default color if ending a line.
    absl::StrAppend(
        &state.line_text,
        UpdateColor(state.color, state.output_format, kDefaultColors));

    int pos = state.line_text.size();
    if (state.source_start != pos && state.source.ptr() != nullptr) {
      state.line_source_map.append(
          nb::make_tuple(state.source_start, pos, state.source));
    }
    state.source_map->append(state.line_source_map);
    state.line_source_map = nb::list();
    state.source_start = next_indent;
  }

  // Transition to default color at line ends if separable or if annotations
  // exist.
  if (state.separable_lines || !state.line_annotations.empty()) {
    absl::StrAppend(
        &state.line_text,
        UpdateColor(state.color, state.output_format, kDefaultColors));
  }

  state.lines.push_back(Line{std::move(state.line_text), state.k,
                             std::move(state.line_annotations)});
}

void NilDoc::Format(const FormatAgendum& agendum, FormatState& state) const {}

void TextDoc::Format(const FormatAgendum& agendum, FormatState& state) const {
  absl::StrAppend(&state.line_text,
                  UpdateColor(state.color, state.output_format, agendum.color));
  if (state.output_format == OutputFormat::kHtml) {
    EscapeHtml(&state.line_text, text_);
  } else {
    absl::StrAppend(&state.line_text, text_);
  }
  if (annotation_.has_value()) {
    state.line_annotations.push_back(*annotation_);
  }
  state.k += text_.size();
}

void ConcatDoc::Format(const FormatAgendum& agendum, FormatState& state) const {
  for (auto it = children_.rbegin(); it != children_.rend(); ++it) {
    state.agenda.push(FormatAgendum{agendum.indent, agendum.mode, it->get(),
                                    agendum.color, state.source});
  }
}

void BreakDoc::Format(const FormatAgendum& agendum, FormatState& state) const {
  if (agendum.mode == BreakMode::kBreak) {
    EndLine(state, agendum.indent);

    state.line_text = std::string(agendum.indent, ' ');
    state.line_annotations.clear();
    state.k = agendum.indent;
  } else {
    absl::StrAppend(
        &state.line_text,
        UpdateColor(state.color, state.output_format, agendum.color));
    if (state.output_format == OutputFormat::kHtml) {
      EscapeHtml(&state.line_text, text_);
    } else {
      absl::StrAppend(&state.line_text, text_);
    }
    state.k += text_.size();
  }
}

void GroupDoc::Format(const FormatAgendum& agendum, FormatState& state) const {
  // In Lindig's paper, _fits is passed the remainder of the document.
  // I'm pretty sure that's a bug and we care only if the current group fits!
  bool fits = ::jax::Fits(agendum.doc, state.width - state.k) &&
              ::jax::Sparse(agendum.doc);
  state.agenda.push(FormatAgendum{agendum.indent,
                                  fits ? BreakMode::kFlat : BreakMode::kBreak,
                                  child_.get(), agendum.color, state.source});
}

void NestDoc::Format(const FormatAgendum& agendum, FormatState& state) const {
  state.agenda.push(FormatAgendum{agendum.indent + n_, agendum.mode,
                                  child_.get(), agendum.color, state.source});
}

void SourceMapDoc::Format(const FormatAgendum& agendum,
                          FormatState& state) const {
  state.agenda.push(FormatAgendum{agendum.indent, agendum.mode, child_.get(),
                                  agendum.color, source_});
}

void ColorDoc::Format(const FormatAgendum& agendum, FormatState& state) const {
  ColorState color = agendum.color;
  if (foreground_.has_value()) {
    color.foreground = *foreground_;
  }
  if (background_.has_value()) {
    color.background = *background_;
  }
  if (intensity_.has_value()) {
    color.intensity = *intensity_;
  }
  state.agenda.push(FormatAgendum{agendum.indent, agendum.mode, child_.get(),
                                  color, state.source});
}

std::string Format(const Doc* doc, int width, bool use_color,
                   OutputFormat output_format, bool separable_lines,
                   std::string annotation_prefix,
                   std::optional<nb::list> source_map) {
  FormatState state;
  if (use_color) {
    state.color = kDefaultColors;
  }
  state.output_format = output_format;
  state.separable_lines = separable_lines;
  state.width = width;
  state.source_start = 0;
  state.source_map = source_map;

  state.agenda.push(
      FormatAgendum{0, BreakMode::kBreak, doc, kDefaultColors, nb::object()});
  state.k = 0;
  while (!state.agenda.empty()) {
    FormatAgendum agendum = state.agenda.top();
    state.agenda.pop();
    if (source_map.has_value() && agendum.source.ptr() != state.source.ptr()) {
      // Transition back to default before recording pos.
      absl::StrAppend(
          &state.line_text,
          UpdateColor(state.color, state.output_format, kDefaultColors));

      int pos = state.line_text.size();
      if (state.source_start != pos && state.source.ptr() != nullptr) {
        state.line_source_map.append(
            nb::make_tuple(state.source_start, pos, state.source));
      }
      state.source = agendum.source;
      state.source_start = pos;
    }
    agendum.doc->Format(agendum, state);
  }

  // Handle the final line.
  EndLine(state, 0);

  int max_width = 0;
  for (const auto& line : state.lines) {
    max_width = std::max(max_width, line.width);
  }

  std::string out = "";

  for (size_t i = 0; i < state.lines.size(); ++i) {
    if (i > 0) {
      absl::StrAppend(&out, "\n");
    }

    const Line& line = state.lines[i];

    absl::StrAppend(&out, line.text);

    for (size_t j = 0; j < line.annotations.size(); ++j) {
      if (j > 0) {
        absl::StrAppend(&out, "\n");
      }

      if (use_color) {
        absl::StrAppend(&out, UpdateColor(state.color, state.output_format,
                                          kAnnotationColors));
      }
      int padding = (j == 0) ? (max_width - line.width) : max_width;
      absl::StrAppend(&out, std::string(padding, ' '), annotation_prefix,
                      line.annotations[j]);
      if (use_color) {
        absl::StrAppend(&out, UpdateColor(state.color, state.output_format,
                                          kDefaultColors));
      }
    }
  }
  if (use_color) {
    absl::StrAppend(
        &out, UpdateColor(state.color, state.output_format, kDefaultColors));
  }
  return out;
}

}  // namespace

NB_MODULE(_pretty_printer, m) {
  nb::enum_<Color>(m, "Color")
      .value("BLACK", Color::kBlack)
      .value("RED", Color::kRed)
      .value("GREEN", Color::kGreen)
      .value("YELLOW", Color::kYellow)
      .value("BLUE", Color::kBlue)
      .value("MAGENTA", Color::kMagenta)
      .value("CYAN", Color::kCyan)
      .value("WHITE", Color::kWhite)
      .value("RESET", Color::kReset);

  nb::enum_<Intensity>(m, "Intensity")
      .value("DIM", Intensity::kDim)
      .value("NORMAL", Intensity::kNormal)
      .value("BRIGHT", Intensity::kBright);

  nb::enum_<OutputFormat>(m, "OutputFormat")
      .value("TEXT", OutputFormat::kText)
      .value("HTML", OutputFormat::kHtml);

  std::string doc_name =
      absl::StrCat(nb::cast<std::string>(m.attr("__name__")), ".Doc");
  PyType_Spec doc_spec = {
      /*.name=*/doc_name.c_str(),
      /*.basicsize=*/static_cast<int>(sizeof(DocObject)),
      /*.itemsize=*/0,
      /*.flags=*/Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
      /*.slots=*/Doc::tp_slots,
  };
  Doc::type = reinterpret_cast<PyTypeObject*>(PyType_FromSpec(&doc_spec));
  if (!Doc::type) {
    throw nb::python_error();
  }
  auto doc_type_obj = nb::borrow<nb::object>(Doc::type);
  m.attr("Doc") = doc_type_obj;

  doc_type_obj.attr("__repr__") = nb::cpp_function(
      [](PyDoc self) { return self->Repr(); }, nb::is_method());

  doc_type_obj.attr("__add__") = nb::cpp_function(
      [](PyDoc self, PyDoc other) -> PyDoc {
        return MakeDoc<ConcatDoc>(std::vector<PyDoc>{self, other});
      },
      nb::is_method(), nb::arg("other"));

  doc_type_obj.attr("_format") = nb::cpp_function(
      [](PyDoc self, int width, bool use_color, OutputFormat output_format,
         bool separable_lines, std::string annotation_prefix,
         std::optional<nb::list> source_map) {
        return Format(self.get(), width, use_color, output_format,
                      separable_lines, annotation_prefix, source_map);
      },
      nb::is_method(), nb::arg("width"), nb::kw_only(), nb::arg("use_color"),
      nb::arg("output_format"), nb::arg("separable_lines"),
      nb::arg("annotation_prefix"), nb::arg("source_map").none());

  // Define subclasses
  auto make_subclass = [&](const char* name_suffix, int size,
                           PyType_Slot* slots) {
    std::string name = absl::StrCat(nb::cast<std::string>(m.attr("__name__")),
                                    ".", name_suffix);
    PyType_Spec spec = {
        /*.name=*/name.c_str(),
        /*.basicsize=*/size,
        /*.itemsize=*/0,
        /*.flags=*/Py_TPFLAGS_DEFAULT,
        /*.slots=*/slots,
    };
    PyObject* type =
        PyType_FromSpecWithBases(&spec, reinterpret_cast<PyObject*>(Doc::type));
    if (!type) {
      throw nb::python_error();
    }
    return reinterpret_cast<PyTypeObject*>(type);
  };

  NilDoc::type =
      make_subclass("NilDoc", sizeof(DocInstance<NilDoc>), NilDoc::tp_slots);
  m.attr("NilDoc") = nb::borrow<nb::object>(NilDoc::type);

  TextDoc::type =
      make_subclass("TextDoc", sizeof(DocInstance<TextDoc>), TextDoc::tp_slots);
  m.attr("TextDoc") = nb::borrow<nb::object>(TextDoc::type);

  ConcatDoc::type = make_subclass("ConcatDoc", sizeof(DocInstance<ConcatDoc>),
                                  ConcatDoc::tp_slots);
  m.attr("ConcatDoc") = nb::borrow<nb::object>(ConcatDoc::type);

  BreakDoc::type = make_subclass("BreakDoc", sizeof(DocInstance<BreakDoc>),
                                 BreakDoc::tp_slots);
  m.attr("BreakDoc") = nb::borrow<nb::object>(BreakDoc::type);

  GroupDoc::type = make_subclass("GroupDoc", sizeof(DocInstance<GroupDoc>),
                                 GroupDoc::tp_slots);
  m.attr("GroupDoc") = nb::borrow<nb::object>(GroupDoc::type);

  NestDoc::type =
      make_subclass("NestDoc", sizeof(DocInstance<NestDoc>), NestDoc::tp_slots);
  m.attr("NestDoc") = nb::borrow<nb::object>(NestDoc::type);

  SourceMapDoc::type =
      make_subclass("SourceMapDoc", sizeof(DocInstance<SourceMapDoc>),
                    SourceMapDoc::tp_slots);
  m.attr("SourceMapDoc") = nb::borrow<nb::object>(SourceMapDoc::type);

  ColorDoc::type = make_subclass("ColorDoc", sizeof(DocInstance<ColorDoc>),
                                 ColorDoc::tp_slots);
  m.attr("ColorDoc") = nb::borrow<nb::object>(ColorDoc::type);

  // Factory functions
  m.def(
      "nil", []() -> PyDoc { return MakeDoc<NilDoc>(); }, "An empty document.");

  m.def(
      "text",
      [](std::string text, std::optional<std::string> annotation) -> PyDoc {
        return MakeDoc<TextDoc>(std::move(text), std::move(annotation));
      },
      nb::arg("text"), nb::arg("annotation").none() = std::nullopt,
      "Literal text.");

  m.def(
      "concat",
      [](std::vector<PyDoc> children) -> PyDoc {
        return MakeDoc<ConcatDoc>(std::move(children));
      },
      nb::arg("children"), "Concatenation of documents.");

  m.def(
      "brk",
      [](std::string text) -> PyDoc {
        return MakeDoc<BreakDoc>(std::move(text));
      },
      nb::arg("text") = std::string(" "),
      R"(A break.

Prints either as a newline or as `text`, depending on the enclosing group.
)");

  m.def(
      "group",
      [](PyDoc child) -> PyDoc { return MakeDoc<GroupDoc>(std::move(child)); },
      R"(Layout alternative groups.

Prints the group with its breaks as their text (typically spaces) if the
entire group would fit on the line when printed that way. Otherwise, breaks
inside the group as printed as newlines.
)");

  m.def(
      "nest",
      [](int n, PyDoc child) -> PyDoc {
        return MakeDoc<NestDoc>(n, std::move(child));
      },
      nb::arg("n"), nb::arg("child"),
      "Increases the indentation level by `n`.");

  m.def(
      "color",
      [](PyDoc child, std::optional<Color> foreground,
         std::optional<Color> background,
         std::optional<Intensity> intensity) -> PyDoc {
        return MakeDoc<ColorDoc>(std::move(child), foreground, background,
                                 intensity);
      },
      nb::arg("child"), nb::arg("foreground").none() = std::nullopt,
      nb::arg("background").none() = std::nullopt,
      nb::arg("intensity").none() = std::nullopt,
      R"(ANSI colors.

Overrides the foreground/background/intensity of the text for the child doc.
Requires use_colors=True to be set when printing; otherwise does nothing.
)");

  m.def(
      "source_map",
      [](PyDoc child, nb::object source) -> PyDoc {
        return MakeDoc<SourceMapDoc>(std::move(child), std::move(source));
      },
      nb::arg("doc"), nb::arg("source"),
      R"(Source mapping.

A source map associates a region of the pretty-printer's text output with a
source location that produced it. For the purposes of the pretty printer a
``source`` may be any object: we require only that we can compare sources for
equality. A text region to source object mapping can be populated as a side
output of the ``format`` method.
)");
}

}  // namespace jax
