#ifndef HOK_HPP
#define HOK_HPP

#include <limits>
#include <type_traits>
#include <sycl/sycl.hpp>

namespace hok {

namespace detail {

template <typename T, typename = void>
struct element {
    using type = T;
    static constexpr int dimensions = 1;
    static constexpr bool is_pointer = 0;
    static constexpr bool is_buffer = 0;
    static constexpr bool is_accessor = 0;
};

template <typename T>
struct element<T*> {
    using type = std::remove_cv_t<T>;
    static constexpr int dimensions = 1;
    static constexpr bool is_pointer = 1;
    static constexpr bool is_buffer = 0;
    static constexpr bool is_accessor = 0;
};

template <typename T>
struct element<const T*> {
    using type = std::remove_cv_t<T>;
    static constexpr int dimensions = 1;
    static constexpr bool is_pointer = 1;
    static constexpr bool is_buffer = 0;
    static constexpr bool is_accessor = 0;
};

template <typename T, int dims, sycl::access::mode access_mode, sycl::access::target access_target, sycl::access::placeholder is_placeholder, typename property_list>
struct element<sycl::accessor<T, dims, access_mode, access_target, is_placeholder, property_list>> {
    using type = T;
    static constexpr int dimensions = dims;
    static constexpr bool is_pointer = 0;
    static constexpr bool is_buffer = 0;
    static constexpr bool is_accessor = 1;
};

template <typename T, int dims, typename allocator_type, typename enable>
struct element<sycl::buffer<T, dims, allocator_type, enable>> {
    using type = T;
    static constexpr int dimensions = dims;
    static constexpr bool is_pointer = 0;
    static constexpr bool is_buffer = 1;
    static constexpr bool is_accessor = 0;
};

template <typename T>
using element_t = typename element<T>::type;

template <typename T>
inline constexpr int element_dimensions = element<T>::dimensions;

template <typename T>
inline constexpr int is_pointer = element<T>::is_pointer;

template <typename T>
inline constexpr int is_buffer = element<T>::is_buffer;

template <typename T>
inline constexpr int is_accessor = element<T>::is_accessor;

template <typename T>
inline constexpr bool is_numeric_v =
    std::numeric_limits<element_t<T>>::is_specialized;

// The data types do not need to be the same or have the same element type, but I did not verified that.
// TODO: is_same_v is necessary?
template <typename T, typename U>
inline constexpr bool is_same_v =
    std::is_same_v<element_t<T>, element_t<U>> &&
    ((is_pointer<T> == is_pointer<U>) ^ (is_buffer<T> == is_buffer<U>) ^ (is_accessor<T> == is_accessor<U>));

}  // namespace detail

// TODO: Refact functors or derive from a common base
// TODO: Deal with image data channels
// TODO: Support unsampled image, if possible
// TODO: Support nd_range

template <
    typename input_type,
    typename output_type,
    int dimensions = detail::element_dimensions<input_type>,
    typename = std::enable_if_t<detail::is_numeric_v<input_type>>,
    typename = std::enable_if_t<detail::is_numeric_v<output_type>>,
    typename = std::enable_if_t<detail::is_same_v<input_type, output_type>>
>
class invert {
public:
    using value_type = detail::element_t<input_type>;

    invert(input_type input_data, output_type output_data)
        : m_input_data(input_data), m_output_data(output_data) {}

    template <typename U = input_type, typename = std::enable_if_t<!detail::is_buffer<U>>>
    void operator()(const sycl::id<dimensions> id) const {
        m_output_data[id] = std::numeric_limits<value_type>::max() - m_input_data[id];
    }

    template <typename U = input_type, typename = std::enable_if_t<!detail::is_accessor<U>>>
    sycl::event submit(const sycl::range<dimensions> range, sycl::queue& queue) {
        if constexpr (detail::is_pointer<input_type>) {
            return queue.parallel_for(range, (*this));
        } else {
            return queue.submit([=] (sycl::handler& handler) {
                auto input_data = m_input_data.get_access(handler, sycl::read_only);
                auto output_data = m_output_data.get_access(handler, sycl::write_only);

                handler.parallel_for(range, hok::invert(input_data, output_data));
            });
        }
    }

private:
    input_type m_input_data;
    output_type m_output_data;
};

template <
    typename input_type,
    typename output_type,
    int dimensions = detail::element_dimensions<input_type>,
    typename = std::enable_if_t<detail::is_numeric_v<input_type>>,
    typename = std::enable_if_t<detail::is_numeric_v<output_type>>,
    typename = std::enable_if_t<detail::is_same_v<input_type, output_type>>
>
class thresh {
public:
    using value_type = detail::element_t<input_type>;

    thresh(input_type input_data, output_type output_data, value_type threshold,
        value_type max_value = std::numeric_limits<value_type>::max(), value_type min_value = std::numeric_limits<value_type>::min())
        : m_input_data(input_data), m_output_data(output_data), m_threshold(threshold), m_max_value(max_value), m_min_value(min_value) {}

    template <typename U = input_type, typename = std::enable_if_t<!detail::is_buffer<U>>>
    void operator()(const sycl::id<dimensions> id) const {
        m_output_data[id] = m_input_data[id] > m_threshold ? m_max_value : m_min_value;
    }

    template <typename U = input_type, typename = std::enable_if_t<!detail::is_accessor<U>>>
    sycl::event submit(const sycl::range<dimensions> range, sycl::queue& queue) {
        if constexpr (detail::is_pointer<input_type>) {
            return queue.parallel_for(range, (*this));
        } else {
            return queue.submit([=] (sycl::handler& handler) {
                auto input_data = m_input_data.get_access(handler, sycl::read_only);
                auto output_data = m_output_data.get_access(handler, sycl::write_only);

                handler.parallel_for(range, hok::thresh(input_data, output_data, m_threshold, m_max_value, m_min_value));
            });
        }
    }

private:
    input_type m_input_data;
    output_type m_output_data;
    value_type m_threshold;
    value_type m_max_value;
    value_type m_min_value;
};

}  // namespace hok

#endif  // HOK_HPP
