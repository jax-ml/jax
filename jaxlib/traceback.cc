/* Copyright 2020 The JAX Authors

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

#include "jaxlib/traceback.h"

#include <Python.h>

#include <array>
#include <atomic>
#include <cstddef>
#include <cstring>
#include <optional>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "absl/base/casts.h"
#include "absl/hash/hash.h"
#include "absl/log/check.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "nanobind/nanobind.h"
#include "nanobind/stl/optional.h"  // IWYU pragma: keep
#include "nanobind/stl/string.h"  // IWYU pragma: keep
#include "nanobind/stl/string_view.h"  // IWYU pragma: keep
#include "nanobind/stl/vector.h"  // IWYU pragma: keep
#include "xla/pjrt/exceptions.h"
#include "xla/python/nb_helpers.h"
#include "tsl/platform/platform.h"

#ifdef PLATFORM_GOOGLE
#define Py_BUILD_CORE
#include "internal/pycore_frame.h"
#undef Py_BUILD_CORE
#endif  // PLATFORM_GOOGLE

namespace nb = nanobind;

namespace xla {

namespace {

std::atomic<bool> traceback_enabled_ = true;

static constexpr int kMaxFrames = 512;

PyTypeObject* traceback_type_ = nullptr;

// Entry in a traceback. Must be POD.
struct TracebackEntry {
  TracebackEntry() = default;
  TracebackEntry(PyCodeObject* code, int lasti) : code(code), lasti(lasti) {}
  PyCodeObject* code;
  int lasti;

  bool operator==(const TracebackEntry& other) const {
    return code == other.code && lasti == other.lasti;
  }
  bool operator!=(const TracebackEntry& other) const {
    return !operator==(other);
  }
};
static_assert(std::is_trivial_v<TracebackEntry> == true);

template <typename H>
H AbslHashValue(H h, const TracebackEntry& entry) {
  h = H::combine(std::move(h), entry.code, entry.lasti);
  return h;
}

struct TracebackObject {
  PyObject_VAR_HEAD;
  TracebackEntry frames[];
};

template <typename H>
H AbslHashValue(H h, const TracebackObject& tb) {
  h = H::combine_contiguous(std::move(h), &tb.frames[0], Py_SIZE(&tb));
  return h;
}

static_assert(sizeof(TracebackObject) % alignof(PyObject) == 0);
static_assert(sizeof(TracebackEntry) % alignof(void*) == 0);

bool traceback_check(nb::handle o) {
  return Py_TYPE(o.ptr()) == traceback_type_;
}

Py_hash_t traceback_tp_hash(PyObject* o) {
  TracebackObject* tb = reinterpret_cast<TracebackObject*>(o);
  size_t h = absl::HashOf(*tb);
  Py_hash_t s = absl::bit_cast<Py_hash_t>(h);  // Python hashes are signed.
  return s == -1 ? -2 : s;  // -1 must not be used as a Python hash value.
}

PyObject* traceback_tp_richcompare(PyObject* self, PyObject* other, int op) {
  if (op != Py_EQ && op != Py_NE) {
    return Py_NewRef(Py_NotImplemented);
  }

  if (!traceback_check(other)) {
    return Py_NewRef(Py_False);
  }
  TracebackObject* tb_self = reinterpret_cast<TracebackObject*>(self);
  TracebackObject* tb_other = reinterpret_cast<TracebackObject*>(other);
  if (Py_SIZE(tb_self) != Py_SIZE(tb_other)) {
    return Py_NewRef(op == Py_EQ ? Py_False : Py_True);
  }
  for (Py_ssize_t i = 0; i < Py_SIZE(tb_self); ++i) {
    if ((tb_self->frames[i] != tb_other->frames[i])) {
      return Py_NewRef(op == Py_EQ ? Py_False : Py_True);
    }
  }
  return Py_NewRef(op == Py_EQ ? Py_True : Py_False);
}

static void traceback_tp_dealloc(PyObject* self) {
  TracebackObject* tb = reinterpret_cast<TracebackObject*>(self);
  for (Py_ssize_t i = 0; i < Py_SIZE(tb); ++i) {
    Py_XDECREF(tb->frames[i].code);
  }
  PyTypeObject* tp = Py_TYPE(self);
  tp->tp_free((PyObject*)self);
  Py_DECREF(tp);
}

Traceback::Frame DecodeFrame(const TracebackEntry& frame) {
  return Traceback::Frame{
      .file_name = nb::borrow<nb::str>(frame.code->co_filename),
      .function_name = nb::borrow<nb::str>(frame.code->co_name),
      .function_start_line = frame.code->co_firstlineno,
      .line_num = PyCode_Addr2Line(frame.code, frame.lasti),
  };
}

std::string traceback_to_string(const TracebackObject* tb) {
  std::vector<std::string> frame_strs;
  frame_strs.reserve(Py_SIZE(tb));
  for (Py_ssize_t i = 0; i < Py_SIZE(tb); ++i) {
    frame_strs.push_back(DecodeFrame(tb->frames[i]).ToString());
  }
  return absl::StrJoin(frame_strs, "\n");
}

PyObject* traceback_tp_str(PyObject* self) {
  TracebackObject* tb = reinterpret_cast<TracebackObject*>(self);
  return nb::cast(traceback_to_string(tb)).release().ptr();
}

// It turns out to be slightly faster to define a tp_hash slot rather than
// defining __hash__ and __eq__ on the class.
PyType_Slot traceback_slots_[] = {
    {Py_tp_hash, reinterpret_cast<void*>(traceback_tp_hash)},
    {Py_tp_richcompare, reinterpret_cast<void*>(traceback_tp_richcompare)},
    {Py_tp_dealloc, reinterpret_cast<void*>(traceback_tp_dealloc)},
    {Py_tp_str, reinterpret_cast<void*>(traceback_tp_str)},
    {0, nullptr},
};

nb::object AsPythonTraceback(const Traceback& tb) {
  nb::object traceback = nb::none();
  nb::dict globals;
  nb::handle traceback_type(reinterpret_cast<PyObject*>(&PyTraceBack_Type));
  TracebackObject* tb_obj = reinterpret_cast<TracebackObject*>(tb.ptr());
  for (Py_ssize_t i = 0; i < Py_SIZE(tb_obj); ++i) {
    const TracebackEntry& frame = tb_obj->frames[i];
    int lineno = PyCode_Addr2Line(frame.code, frame.lasti);
    // Under Python 3.11 we observed crashes when using a fake PyFrameObject
    // with a real PyCodeObject (https://github.com/google/jax/issues/16027).
    // because the frame does not have fields necessary to compute the locals,
    // notably the closure object, leading to crashes in CPython in
    // _PyFrame_FastToLocalsWithError
    // https://github.com/python/cpython/blob/deaf509e8fc6e0363bd6f26d52ad42f976ec42f2/Objects/frameobject.c#LL1116C2-L1116C2
    // We therefore always build a fake code object to go along with our fake
    // frame.
    PyCodeObject* py_code =
        PyCode_NewEmpty(PyUnicode_AsUTF8(frame.code->co_filename),
                        PyUnicode_AsUTF8(frame.code->co_name), lineno);
    PyFrameObject* py_frame = PyFrame_New(PyThreadState_Get(), py_code,
                                          globals.ptr(), /*locals=*/nullptr);
    Py_DECREF(py_code);

    traceback = traceback_type(
        /*tb_next=*/std::move(traceback),
        /*tb_frame=*/
        nb::steal<nb::object>(reinterpret_cast<PyObject*>(py_frame)),
        /*tb_lasti=*/0,
        /*tb_lineno=*/lineno);
  }
  return traceback;
}

}  // namespace

std::vector<Traceback::Frame> Traceback::Frames() const {
  // We require the GIL because we manipulate Python strings.
  CHECK(PyGILState_Check());
  std::vector<Traceback::Frame> frames;
  TracebackObject* tb = reinterpret_cast<TracebackObject*>(ptr());
  frames.reserve(Py_SIZE(tb));
  for (Py_ssize_t i = 0; i < Py_SIZE(tb); ++i) {
    const TracebackEntry& frame = tb->frames[i];
    frames.push_back(Frame{nb::borrow<nb::str>(frame.code->co_filename),
                           nb::borrow<nb::str>(frame.code->co_name),
                           frame.code->co_firstlineno,
                           PyCode_Addr2Line(frame.code, frame.lasti)});
  }
  return frames;
}

std::string Traceback::Frame::ToString() const {
  return absl::StrFormat("%s:%d (%s)", nb::cast<absl::string_view>(file_name),
                         line_num, nb::cast<absl::string_view>(function_name));
}

std::string Traceback::ToString() const {
  return traceback_to_string(reinterpret_cast<const TracebackObject*>(ptr()));
}

std::vector<std::pair<PyCodeObject*, int>> Traceback::RawFrames() const {
  const TracebackObject* tb = reinterpret_cast<const TracebackObject*>(ptr());
  std::vector<std::pair<PyCodeObject*, int>> frames;
  frames.reserve(Py_SIZE(tb));
  for (Py_ssize_t i = 0; i < Py_SIZE(tb); ++i) {
    frames.push_back(std::make_pair(tb->frames[i].code, tb->frames[i].lasti));
  }
  return frames;
}

/*static*/ bool Traceback::Check(PyObject* o) { return traceback_check(o); }

/*static*/ std::optional<Traceback> Traceback::Get() {
  // We use a thread_local here mostly to avoid requiring a large amount of
  // space.
  thread_local std::array<TracebackEntry, kMaxFrames> frames;
  int count = 0;

  DCHECK(PyGILState_Check());

  if (!traceback_enabled_.load()) {
    return std::nullopt;
  }

  PyThreadState* thread_state = PyThreadState_GET();

#ifdef PLATFORM_GOOGLE
// This code is equivalent to the version using public APIs, but it saves us
// an allocation of one object per stack frame. However, this is definitely
// violating the API contract of CPython, so we only use this where we can be
// confident we know exactly which CPython we are using (internal to Google).
// Feel free to turn this on if you like, but it might break at any time!
#if PY_VERSION_HEX < 0x030d0000
  for (_PyInterpreterFrame* f = thread_state->cframe->current_frame;
       f != nullptr && count < kMaxFrames; f = f->previous) {
    if (_PyFrame_IsIncomplete(f)) continue;
    Py_INCREF(f->f_code);
    frames[count] = {f->f_code, static_cast<int>(_PyInterpreterFrame_LASTI(f) *
                                                 sizeof(_Py_CODEUNIT))};
    ++count;
  }
#else   // PY_VERSION_HEX < 0x030d0000
  for (_PyInterpreterFrame* f = thread_state->current_frame;
       f != nullptr && count < kMaxFrames; f = f->previous) {
    if (_PyFrame_IsIncomplete(f)) continue;
    Py_INCREF(f->f_executable);
    frames[count] = {
        reinterpret_cast<PyCodeObject*>(f->f_executable),
        static_cast<int>(_PyInterpreterFrame_LASTI(f) * sizeof(_Py_CODEUNIT))};
    ++count;
  }
#endif  // PY_VERSION_HEX < 0x030d0000

#else   // PLATFORM_GOOGLE
  PyFrameObject* next;
  for (PyFrameObject* py_frame = PyThreadState_GetFrame(thread_state);
       py_frame != nullptr && count < kMaxFrames; py_frame = next) {
    frames[count] = {PyFrame_GetCode(py_frame), PyFrame_GetLasti(py_frame)};
    ++count;
    next = PyFrame_GetBack(py_frame);
    Py_XDECREF(py_frame);
  }
#endif  // PLATFORM_GOOGLE

  Traceback traceback =
      nb::steal<Traceback>(PyObject_NewVar(PyObject, traceback_type_, count));
  TracebackObject* tb = reinterpret_cast<TracebackObject*>(traceback.ptr());
  std::memcpy(tb->frames, frames.data(), sizeof(TracebackEntry) * count);
  return traceback;
}

void BuildTracebackSubmodule(nb::module_& m) {
  nb::class_<Traceback::Frame>(m, "Frame")
      .def(nb::init<const nb::str&, const nb::str&, int, int>())
      .def_ro("file_name", &Traceback::Frame::file_name)
      .def_ro("function_name", &Traceback::Frame::function_name)
      .def_ro("function_start_line", &Traceback::Frame::function_start_line)
      .def_ro("line_num", &Traceback::Frame::line_num)
      .def("__repr__", [](const Traceback::Frame& frame) {
        return absl::StrFormat(
            "%s;%s:%d", nb::cast<absl::string_view>(frame.function_name),
            nb::cast<absl::string_view>(frame.file_name), frame.line_num);
      });

  std::string name =
      absl::StrCat(nb::cast<std::string>(m.attr("__name__")), ".Traceback");

  PyType_Spec traceback_spec = {
      /*.name=*/name.c_str(),
      /*.basicsize=*/static_cast<int>(sizeof(TracebackObject)),
      /*.itemsize=*/static_cast<int>(sizeof(TracebackEntry)),
      /*.flags=*/Py_TPFLAGS_DEFAULT,
      /*.slots=*/traceback_slots_,
  };

  traceback_type_ =
      reinterpret_cast<PyTypeObject*>(PyType_FromSpec(&traceback_spec));
  if (!traceback_type_) {
    throw nb::python_error();
  }

  auto type = nb::borrow<nb::object>(traceback_type_);
  m.attr("Traceback") = type;

  m.def("tracebacks_enabled", []() { return traceback_enabled_.load(); });
  m.def("set_tracebacks_enabled",
        [](bool value) { traceback_enabled_.store(value); });

  type.attr("get_traceback") = nb::cpp_function(Traceback::Get,
                                                R"doc(
      Returns a :class:`Traceback` for the current thread.

      If ``Traceback.enabled`` is ``True``, returns a :class:`Traceback`
      object that describes the Python stack of the calling thread. Stack
      trace collection has a small overhead, so it is disabled by default. If
      traceback collection is disabled, returns ``None``. )doc");
  type.attr("frames") = nb_property_readonly(&Traceback::Frames);
  type.attr("raw_frames") = nb::cpp_function(
      [](const Traceback& tb) -> nb::tuple {
        // We return a tuple of lists, rather than a list of tuples, because it
        // is cheaper to allocate only three Python objects for everything
        // rather than one per frame.
        std::vector<std::pair<PyCodeObject*, int>> frames = tb.RawFrames();
        nb::list out_code = nb::steal<nb::list>(PyList_New(frames.size()));
        nb::list out_lasti = nb::steal<nb::list>(PyList_New(frames.size()));
        for (size_t i = 0; i < frames.size(); ++i) {
          const auto& frame = frames[i];
          PyObject* code = reinterpret_cast<PyObject*>(frame.first);
          Py_INCREF(code);
          PyList_SET_ITEM(out_code.ptr(), i, code);
          PyList_SET_ITEM(out_lasti.ptr(), i,
                          nb::int_(frame.second).release().ptr());
        }
        return nb::make_tuple(out_code, out_lasti);
      },
      nb::is_method());
  type.attr("as_python_traceback") =
      nb::cpp_function(AsPythonTraceback, nb::is_method());

  type.attr("traceback_from_frames") = nb::cpp_function(
      [](std::vector<Traceback::Frame> frames) {
        nb::object traceback = nb::none();
        nb::dict globals;
        nb::handle traceback_type(
            reinterpret_cast<PyObject*>(&PyTraceBack_Type));
        for (const Traceback::Frame& frame : frames) {
          PyCodeObject* py_code =
              PyCode_NewEmpty(frame.file_name.c_str(),
                              frame.function_name.c_str(), frame.line_num);
          PyFrameObject* py_frame = PyFrame_New(PyThreadState_Get(), py_code,
                                                globals.ptr(), /*locals=*/
                                                nullptr);
          Py_DECREF(py_code);
          traceback = traceback_type(
              /*tb_next=*/std::move(traceback),
              /*tb_frame=*/
              nb::steal<nb::object>(reinterpret_cast<PyObject*>(py_frame)),
              /*tb_lasti=*/0,
              /*tb_lineno=*/
              frame.line_num);
        }
        return traceback;
      },
      "Creates a traceback from a list of frames.");

  type.attr("code_addr2line") = nb::cpp_function(
      [](nb::handle code, int lasti) {
        if (!PyCode_Check(code.ptr())) {
          throw xla::XlaRuntimeError("code argument must be a code object");
        }
        return PyCode_Addr2Line(reinterpret_cast<PyCodeObject*>(code.ptr()),
                                lasti);
      },
      "Python wrapper around the Python C API function PyCode_Addr2Line");

  type.attr("code_addr2location") = nb::cpp_function(
      [](nb::handle code, int lasti) {
        if (!PyCode_Check(code.ptr())) {
          throw xla::XlaRuntimeError("code argument must be a code object");
        }
        int start_line, start_column, end_line, end_column;
        if (!PyCode_Addr2Location(reinterpret_cast<PyCodeObject*>(code.ptr()),
                                  lasti, &start_line, &start_column, &end_line,
                                  &end_column)) {
          throw nb::python_error();
        }
        return nb::make_tuple(start_line, start_column, end_line, end_column);
      },
      "Python wrapper around the Python C API function PyCode_Addr2Location");
}
}  // namespace xla
