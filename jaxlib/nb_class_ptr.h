/* Copyright 2024 The JAX Authors

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

#ifndef JAXLIB_NB_CLASS_PTR_H_
#define JAXLIB_NB_CLASS_PTR_H_

#include <type_traits>

#include "nanobind/nanobind.h"

namespace jax {

// A reference-counting smart pointer to a nanobind-wrapped class on the Python
// heap. Type T must be a class known to nanobind via a nanobind::class_
// declaration. nb_class_ptr is useful for managing C++ classes that may be
// allocated inline in Python objects on the Python heap.
template <typename T>
class nb_class_ptr : public nanobind::object {
 public:
  static constexpr auto Name = nanobind::detail::make_caster<T>::Name;

  inline nb_class_ptr() : nanobind::object() {}
  inline nb_class_ptr(nanobind::handle h, ::nanobind::detail::borrow_t)
      : nanobind::object(h, ::nanobind::detail::borrow_t{}) {}
  inline nb_class_ptr(nanobind::handle h, ::nanobind::detail::steal_t)
      : nanobind::object(h, ::nanobind::detail::steal_t{}) {}
  inline static bool check_(nanobind::handle h) {
    nanobind::handle type = nanobind::type<T>();
    return nanobind::isinstance(h, type);
  };

  template <typename U,
            typename = std::enable_if_t<std::is_convertible_v<U*, T*>>>
  inline nb_class_ptr(nb_class_ptr<U>&& other)
      : nanobind::object(other.release(), ::nanobind::detail::steal_t{}) {}

  T* operator->() const { return nanobind::inst_ptr<T>(ptr()); }
  T& operator*() const { return *nanobind::inst_ptr<T>(ptr()); }
  T* get() const { return ptr() ? nanobind::inst_ptr<T>(ptr()) : nullptr; }
};

// This function is analogous to std::make_unique<T>(...), but instead it
// allocates the object on the Python heap
template <typename T, class... Args>
nb_class_ptr<T> make_nb_class(Args&&... args) {
  nanobind::handle type = nanobind::type<T>();
  nanobind::object instance = nanobind::inst_alloc(type);
  T* ptr = nanobind::inst_ptr<T>(instance);
  new (ptr) T(std::forward<Args>(args)...);
  nanobind::inst_mark_ready(instance);
  return nb_class_ptr<T>(instance.release(), ::nanobind::detail::steal_t{});
}

}  // namespace jax

#endif  //  JAXLIB_NB_CLASS_PTR_H_
