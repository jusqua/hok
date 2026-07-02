#ifndef HOK_HPP
#define HOK_HPP

#include <limits>
#include <type_traits>
#include <sycl/sycl.hpp>

namespace hok {

namespace util {

// TODO: find a better naming for those channel related operations

/// Get a channel range stride
template <int dims>
inline sycl::range<dims> channels_stride(size_t channels) {
    if constexpr (dims == 3) {
        return sycl::range<dims>(1, 1, channels);
    } if constexpr (dims == 2) {
        return sycl::range<dims>(1, channels);
    } else {
        return sycl::range<dims>(channels);
    }
}

/// Get a channel range offset
template <int dims>
inline sycl::range<dims> channels_offset(size_t offset) {
    if constexpr (dims == 3) {
        return sycl::range<dims>(0, 0, offset);
    } if constexpr (dims == 2) {
        return sycl::range<dims>(0, offset);
    } else {
        return sycl::range<dims>(offset);
    }
}

/// Bake channel in the last dimension
template <int dimensions>
inline sycl::range<dimensions> channeled_range(sycl::range<dimensions> range, size_t channels) {
    return range * channels_stride<dimensions>(channels);
}

} // namespace util

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

template <
    typename input_type,
    typename output_type,
    int dimensions = detail::element_dimensions<input_type>,
    typename = std::enable_if_t<detail::is_numeric_v<input_type>>,
    typename = std::enable_if_t<detail::is_numeric_v<output_type>>,
    typename = std::enable_if_t<detail::is_same_v<input_type, output_type>>
>
class gray {
public:
    using value_type = detail::element_t<input_type>;

    gray(input_type input_data, output_type output_data, std::size_t channels)
        : m_input_data(input_data), m_output_data(output_data), m_channels(channels) {}

    template <typename U = input_type, typename = std::enable_if_t<!detail::is_buffer<U>>>
    void operator()(const sycl::id<dimensions> id) const {
        auto cid = id * util::channels_stride<dimensions>(m_channels);
        auto r = m_input_data[cid];
        auto g = m_input_data[cid + util::channels_offset<dimensions>(1)];
        auto b = m_input_data[cid + util::channels_offset<dimensions>(2)];

        // BT.601 color space gray conversion
        value_type gray;
        if constexpr (std::is_integral_v<value_type>)
            gray = (r * 9798 + g * 19235 + b * 3735 + 16384) >> 15;
        else if constexpr (std::is_floating_point_v<value_type>)
            gray = r * 0.299f + g * 0.587f + b * 0.114f;
        else
            static_assert(false, "value_type must be an integral or floating-point type");

        // TODO: output data may have smaller channels than the input data
        for (size_t i = 0; i < 3; i++)
            m_output_data[cid + util::channels_offset<dimensions>(i)] = gray;
    }

    template <typename U = input_type, typename = std::enable_if_t<!detail::is_accessor<U>>>
    sycl::event submit(const sycl::range<dimensions> range, sycl::queue& queue) {
        if constexpr (detail::is_pointer<input_type>) {
            return queue.parallel_for(range, (*this));
        } else {
            return queue.submit([=] (sycl::handler& handler) {
                auto input_data = m_input_data.get_access(handler, sycl::read_only);
                auto output_data = m_output_data.get_access(handler, sycl::write_only);

                handler.parallel_for(range, hok::gray(input_data, output_data, m_channels));
            });
        }
    }

private:
    input_type m_input_data;
    output_type m_output_data;
    std::size_t m_channels;
};

}  // namespace hok

#endif  // HOK_HPP
