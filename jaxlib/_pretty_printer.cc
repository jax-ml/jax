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
#include <optional>
#include <stack>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/inlined_vector.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/types/span.h"
#include "nanobind/nanobind.h"
#include "nanobind/stl/optional.h"  // IWYU pragma: keep
#include "nanobind/stl/string.h"  // IWYU pragma: keep
#include "nanobind/stl/vector.h"  // IWYU pragma: keep
#include "jaxlib/nb_class_ptr.h"

namespace nb = nanobind;

namespace jax {

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

 private:
  int num_annotations_;
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

 private:
  std::string text_;
  std::optional<std::string> annotation_;
};

class ConcatDoc final : public Doc {
 public:
  explicit ConcatDoc(std::vector<xla::nb_class_ptr<Doc>> children)
      : Doc(TotalNumAnnotations(children)), children_(std::move(children)) {}
  std::string Repr() const override;

  void Fits(std::stack<const Doc*>& agenda, int& width) const override;
  bool Sparse(std::stack<const Doc*>& agenda, int& num_annotations,
              bool& seen_break) const override;
  virtual void Format(const FormatAgendum& agendum,
                      FormatState& state) const override;

 private:
  static int TotalNumAnnotations(
      absl::Span<const xla::nb_class_ptr<Doc>> children) {
    int total = 0;
    for (const auto& child : children) {
      total += child->num_annotations();
    }
    return total;
  }
  std::vector<xla::nb_class_ptr<Doc>> children_;
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

 private:
  std::string text_;
};

class GroupDoc final : public Doc {
 public:
  explicit GroupDoc(xla::nb_class_ptr<Doc> child)
      : Doc(/*num_annotations=*/child->num_annotations()),
        child_(std::move(child)) {}
  std::string Repr() const override;
  void Fits(std::stack<const Doc*>& agenda, int& width) const override;
  bool Sparse(std::stack<const Doc*>& agenda, int& num_annotations,
              bool& seen_break) const override;
  virtual void Format(const FormatAgendum& agendum,
                      FormatState& state) const override;

 private:
  xla::nb_class_ptr<Doc> child_;
};

class NestDoc final : public Doc {
 public:
  explicit NestDoc(int n, xla::nb_class_ptr<Doc> child)
      : Doc(child->num_annotations()), n_(n), child_(std::move(child)) {}
  std::string Repr() const override;
  void Fits(std::stack<const Doc*>& agenda, int& width) const override;
  bool Sparse(std::stack<const Doc*>& agenda, int& num_annotations,
              bool& seen_break) const override;
  virtual void Format(const FormatAgendum& agendum,
                      FormatState& state) const override;

 private:
  int n_;
  xla::nb_class_ptr<Doc> child_;
};

class SourceMapDoc final : public Doc {
 public:
  explicit SourceMapDoc(xla::nb_class_ptr<Doc> child, nb::object source)
      : Doc(child->num_annotations()),
        child_(std::move(child)),
        source_(std::move(source)) {}
  std::string Repr() const override;
  void Fits(std::stack<const Doc*>& agenda, int& width) const override;
  bool Sparse(std::stack<const Doc*>& agenda, int& num_annotations,
              bool& seen_break) const override;
  virtual void Format(const FormatAgendum& agendum,
                      FormatState& state) const override;

 private:
  xla::nb_class_ptr<Doc> child_;
  nb::object source_;
};

class ColorDoc final : public Doc {
 public:
  explicit ColorDoc(xla::nb_class_ptr<Doc> child,
                    std::optional<Color> foreground,
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

 private:
  xla::nb_class_ptr<Doc> child_;
  std::optional<Color> foreground_;
  std::optional<Color> background_;
  std::optional<Intensity> intensity_;
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
};

std::string UpdateColor(std::optional<ColorState>& state,
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

void NilDoc::Format(const FormatAgendum& agendum, FormatState& state) const {}

void TextDoc::Format(const FormatAgendum& agendum, FormatState& state) const {
  absl::StrAppend(&state.line_text, UpdateColor(state.color, agendum.color),
                  text_);
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
    if (!state.line_annotations.empty()) {
      absl::StrAppend(&state.line_text,
                      UpdateColor(state.color, kAnnotationColors));
    }
    if (state.source_map.has_value()) {
      int pos = state.line_text.size();
      if (state.source_start != pos && state.source.ptr() != nullptr) {
        state.line_source_map.append(
            nb::make_tuple(state.source_start, pos, state.source));
      }
      state.source_map->append(state.line_source_map);
      state.line_source_map = nb::list();
      state.source_start = agendum.indent;
    }
    state.lines.push_back(Line{std::move(state.line_text), state.k,
                               std::move(state.line_annotations)});
    state.line_text = std::string(agendum.indent, ' ');
    state.line_annotations.clear();
    state.k = agendum.indent;
  } else {
    absl::StrAppend(&state.line_text, UpdateColor(state.color, agendum.color),
                    text_);
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
                   std::string annotation_prefix,
                   std::optional<nb::list> source_map) {
  FormatState state;
  if (use_color) {
    state.color = kDefaultColors;
  }
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
  if (!state.line_annotations.empty()) {
    absl::StrAppend(&state.line_text,
                    UpdateColor(state.color, kAnnotationColors));
  }
  if (state.source_map.has_value()) {
    int pos = state.line_text.size();
    if (state.source_start != pos && state.source.ptr() != nullptr) {
      state.line_source_map.append(
          nb::make_tuple(state.source_start, pos, state.source));
    }
    state.source_map->append(state.line_source_map);
  }
  state.lines.push_back(Line{std::move(state.line_text), state.k,
                             std::move(state.line_annotations)});

  int max_width = 0;
  for (const auto& line : state.lines) {
    max_width = std::max(max_width, line.width);
  }
  std::string out =
      absl::StrJoin(state.lines, "\n", [&](std::string* out, const Line& line) {
        if (line.annotations.empty()) {
          absl::StrAppend(out, line.text);
        } else {
          absl::StrAppend(out, line.text,
                          std::string(max_width - line.width, ' '),
                          annotation_prefix, line.annotations[0]);
          for (int i = 1; i < line.annotations.size(); ++i) {
            absl::StrAppend(out, std::string(max_width, ' '), annotation_prefix,
                            line.annotations[i]);
          }
        }
      });
  absl::StrAppend(&out, UpdateColor(state.color, kDefaultColors));
  return out;
}

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

  nb::class_<Doc>(m, "Doc")
      .def("__repr__", &Doc::Repr)
      .def("__add__",
           [](xla::nb_class_ptr<Doc> self, xla::nb_class_ptr<Doc> other) {
             return xla::make_nb_class<ConcatDoc>(
                 std::vector<xla::nb_class_ptr<Doc>>{std::move(self),
                                                     std::move(other)});
           })
      .def("_format", &Format, nb::arg("width"), nb::arg("use_color"),
           nb::arg("annotation_prefix"), nb::arg("source_map").none());

  nb::class_<NilDoc, Doc>(m, "NilDoc");
  nb::class_<TextDoc, Doc>(m, "TextDoc");
  nb::class_<ConcatDoc, Doc>(m, "ConcatDoc");
  nb::class_<BreakDoc, Doc>(m, "BreakDoc");
  nb::class_<GroupDoc, Doc>(m, "GroupDoc");
  nb::class_<NestDoc, Doc>(m, "NestDoc");
  nb::class_<ColorDoc, Doc>(m, "ColorDoc");
  nb::class_<SourceMapDoc, Doc>(m, "SourceMapDoc");

  m.def(
      "nil", []() { return xla::make_nb_class<NilDoc>(); },
      "An empty document.");
  m.def(
      "text",
      [](std::string text, std::optional<std::string> annotation) {
        return xla::make_nb_class<TextDoc>(std::move(text),
                                           std::move(annotation));
      },
      nb::arg("text"), nb::arg("annotation").none() = std::nullopt,
      "Literal text.");
  m.def(
      "concat",
      [](std::vector<xla::nb_class_ptr<Doc>> children) {
        return xla::make_nb_class<ConcatDoc>(std::move(children));
      },
      nb::arg("children"), "Concatenation of documents.");
  m.def(
      "brk",
      [](std::string text) { return xla::make_nb_class<BreakDoc>(text); },
      nb::arg("text") = std::string(" "),
      R"(A break.

Prints either as a newline or as `text`, depending on the enclosing group.
)");
  m.def(
      "group",
      [](xla::nb_class_ptr<Doc> child) {
        return xla::make_nb_class<GroupDoc>(std::move(child));
      },
      R"(Layout alternative groups.

Prints the group with its breaks as their text (typically spaces) if the
entire group would fit on the line when printed that way. Otherwise, breaks
inside the group as printed as newlines.
)");
  m.def(
      "nest",
      [](int n, xla::nb_class_ptr<Doc> child) {
        return xla::make_nb_class<NestDoc>(n, std::move(child));
      },
      "Increases the indentation level by `n`.");
  m.def(
      "color",
      [](xla::nb_class_ptr<Doc> child, std::optional<Color> foreground,
         std::optional<Color> background, std::optional<Intensity> intensity) {
        return xla::make_nb_class<ColorDoc>(std::move(child), foreground,
                                            background, intensity);
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
      [](xla::nb_class_ptr<Doc> child, nb::object source) {
        return xla::make_nb_class<SourceMapDoc>(std::move(child),
                                                std::move(source));
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
