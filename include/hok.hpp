#ifndef HOK_HPP
#define HOK_HPP

#include <cstdint>
#include <limits>
#include <type_traits>
#include <utility>
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

template <int Dims, typename F>
inline constexpr void nested_loop(sycl::range<Dims> range, F&& func, sycl::id<Dims> indices = {}) {
    if constexpr (Dims == 1) {
        for (auto i = 0; i < range[0]; ++i) {
            indices[0] = i;
            std::forward<F>(func)(indices);
        }
    } else if constexpr (Dims == 2) {
        for (auto i = 0; i < range[0]; ++i) {
            indices[0] = i;
            for (auto j = 0; j < range[1]; ++j) {
                indices[1] = j;
                std::forward<F>(func)(indices);
            }
        }
    } else if constexpr (Dims == 3) {
        for (auto i = 0; i < range[0]; ++i) {
            indices[0] = i;
            for (auto j = 0; j < range[1]; ++j) {
                indices[1] = j;
                for (auto k = 0; k < range[1]; ++k) {
                    indices[2] = k;
                    std::forward<F>(func)(indices);
                }
            }
        }
    } else {
        static_assert(false, "ND not implemented yet");
    }
}

template <int Dims>
inline constexpr auto get_channel_stride_id(sycl::id<Dims> id, int channels) {
    if constexpr (Dims == 3) {
        return id * sycl::range<Dims>(1, 1, channels);
    } if constexpr (Dims == 2) {
        return id * sycl::range<Dims>(1, channels);
    } else {
        return id * sycl::range<Dims>(channels);
    }
}

template <int Dims>
inline constexpr auto get_channel_offset_id(size_t offset) {
    if constexpr (Dims == 3) {
        return sycl::id<Dims>(0, 0, offset);
    } if constexpr (Dims == 2) {
        return sycl::id<Dims>(0, offset);
    } else {
        return sycl::id<Dims>(offset);
    }
}

template <size_t... Is>
constexpr auto to_range_impl(const size_t& value, std::index_sequence<Is...>) {
    return sycl::range<sizeof...(Is)>{ (void(Is), value)... };
}

template <size_t... Is>
constexpr auto to_id_impl(const size_t& value, std::index_sequence<Is...>) {
    return sycl::id<sizeof...(Is)>{ (void(Is), value)... };
}

template <typename DataT, int Dims>
inline constexpr auto read(DataT data, sycl::id<Dims> index, int channels) {
    using T = element_t<DataT>;
    auto result = sycl::float4{0.0f};

    if constexpr (std::is_same_v<T, float>) {
        for (auto i = 0; i < channels; i++) {
            result[i] = data[index + get_channel_offset_id<Dims>(i)];
        }
    } else {
        static_assert(std::is_same_v<T, uint8_t>, "At least for now, must be uint8_t");

        for (auto i = 0; i < channels; ++i) {
            result[i] = static_cast<float>(data[index + get_channel_offset_id<Dims>(i)]) / std::numeric_limits<T>::max();
        }
    }

    return result;
}

template <typename DataT, int Dims>
inline constexpr auto write(DataT raw, sycl::id<Dims> id, int channels, sycl::float4 vec) {
    using T = element_t<DataT>;
    if constexpr (std::is_same_v<T, float>) {
        for (auto i = 0; i < channels; i++) {
            raw[id + detail::get_channel_offset_id<Dims>(i)] = vec[i];
        }
    } else {
        static_assert(std::is_same_v<T, uint8_t>, "At least for now, must be uint8_t");

        for (auto i = 0; i < channels; i++) {
            raw[id + detail::get_channel_offset_id<Dims>(i)] = static_cast<T>(sycl::floor(vec[i] * std::numeric_limits<T>::max()));
        }
    }
}

// TODO: find a better naming for those channel related operations
/// Get a channel range stride
template <int Dims>
inline sycl::range<Dims> channels_stride(size_t channels) {
    if constexpr (Dims == 3) {
        return sycl::range<Dims>(1, 1, channels);
    } if constexpr (Dims == 2) {
        return sycl::range<Dims>(1, channels);
    } else {
        return sycl::range<Dims>(channels);
    }
}

/// Get a channel range offset
template <int Dims>
auto channels_offset(size_t offset) {
    if constexpr (Dims == 3) {
        return sycl::range<Dims>(0, 0, offset);
    } if constexpr (Dims == 2) {
        return sycl::range<Dims>(0, offset);
    } else {
        return sycl::range<Dims>(offset);
    }
}

/// Bake channel in the last dimension
template <int Dims>
auto channeled_range(sycl::range<Dims> range, size_t channels) {
    return range * channels_stride<Dims>(channels);
}

template<int Dims>
auto to_range(size_t value) {
    return detail::to_range_impl(value, std::make_index_sequence<Dims>{});
}

template<int Dims>
auto to_id(size_t value) {
    return detail::to_id_impl(value, std::make_index_sequence<Dims>{});
}

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

    gray(input_type input_data, output_type output_data, size_t channels)
        : m_input_data(input_data), m_output_data(output_data), m_channels(channels) {}

    template <typename U = input_type, typename = std::enable_if_t<!detail::is_buffer<U>>>
    void operator()(const sycl::id<dimensions> id) const {
        auto cid = id * detail::channels_stride<dimensions>(m_channels);
        auto r = m_input_data[cid];
        auto g = m_input_data[cid + detail::channels_offset<dimensions>(1)];
        auto b = m_input_data[cid + detail::channels_offset<dimensions>(2)];

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
            m_output_data[cid + detail::channels_offset<dimensions>(i)] = gray;
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
    size_t m_channels;
};

template <int dimensions>
class convolve {
public:
    convolve(uint8_t* input_data, uint8_t* output_data, int channels, float* filter_data, sycl::range<dimensions> filter_extent)
        : m_input_data(input_data), m_output_data(output_data), m_channels(channels),
          m_filter_data(filter_data), m_filter_extent(filter_extent), m_filter_halo(vec(filter_extent) / 2) {}

    void operator()(const sycl::item<dimensions> item) const {
        auto px = sycl::float4{0};
        map(m_filter_extent, [&](sycl::id<dimensions> fid) {
            px += read(m_input_data, get_linear_id(item, vec(fid) - m_filter_halo))
                * m_filter_data[get_linear_id(m_filter_extent, fid)];
        });
        write(m_output_data, item, px);
    }

private:
    uint8_t* m_input_data;
    uint8_t* m_output_data;
    int m_channels;
    float* m_filter_data;
    sycl::range<dimensions> m_filter_extent;
    sycl::vec<int, dimensions> m_filter_halo;

    inline constexpr auto get_linear_id(const sycl::range<dimensions>& extent, const sycl::id<dimensions>& index) const {
        size_t id = 0;
        if constexpr (dimensions == 1) {
          id = index[0];
        } else if constexpr (dimensions == 2) {
          id = index[0] * extent[1] + index[1];
        } else if constexpr (dimensions == 3) {
          id = index[0] * extent[1] * extent[2] + index[1] * extent[2] + index[2];
        } else {
            static_assert(false, "ND not implemented yet");
        }
        return id;
    }

    inline constexpr auto get_linear_id(const sycl::item<dimensions>& item, const sycl::vec<int, dimensions>& displacement) const {
        sycl::id<dimensions> index = item.get_id();
        sycl::range<dimensions> extent = item.get_range();

        if constexpr (dimensions > 3) {
            static_assert(false, "ND not implemented yet");
        }
        if constexpr (dimensions > 0) {
            index[0] = sycl::clamp(static_cast<int>(index[0]) + displacement[0], 0, static_cast<int>(extent[0]) - 1);
        }
        if constexpr (dimensions > 1) {
            index[1] = sycl::clamp(static_cast<int>(index[1]) + displacement[1], 0, static_cast<int>(extent[1]) - 1);
        }
        if constexpr (dimensions > 2) {
            index[2] = sycl::clamp(static_cast<int>(index[2]) + displacement[2], 0, static_cast<int>(extent[2]) - 1);
        }

        return get_linear_id(extent, index);
    }

    inline constexpr auto get_linear_id(const sycl::item<dimensions>& item) const {
        return get_linear_id(item.get_range(), item.get_id());
    }

    inline constexpr auto read(const uint8_t* data, size_t index) const {
        auto value = sycl::float4{0.0f};
        for (auto i = 0; i < m_channels; i++) {
            value[i] = static_cast<float>(data[index * m_channels + i]) / std::numeric_limits<uint8_t>::max();
        }
        return value;
    }

    inline constexpr auto read(uint8_t* data, const sycl::item<dimensions>& item) const {
        return read(data, item.get_linear_id());
    }

    inline constexpr auto write(uint8_t* data, size_t index, const sycl::float4& value) const {
        for (auto i = 0; i < m_channels; i++) {
            data[index * m_channels + i] = static_cast<uint8_t>(sycl::trunc(sycl::clamp(value[i], 0.0f, 1.0f) * std::numeric_limits<uint8_t>::max()));
        }
    }

    inline constexpr auto write(uint8_t* data, const sycl::item<dimensions>& item, const sycl::float4& value) const {
        return write(data, item.get_linear_id(), value);
    }

    template <typename F>
    inline constexpr auto map(sycl::range<dimensions> range, F&& func) const {
        if constexpr (dimensions == 1) {
            for (size_t i = 0; i < range[0]; ++i) {
                std::forward<F>(func)(sycl::id<dimensions>{i});
            }
        } else if constexpr (dimensions == 2) {
            for (size_t i = 0; i < range[0]; ++i) {
                for (size_t j = 0; j < range[1]; ++j) {
                    std::forward<F>(func)(sycl::id<dimensions>{i, j});
                }
            }
        } else if constexpr (dimensions == 3) {
            for (size_t i = 0; i < range[0]; ++i) {
                for (size_t j = 0; j < range[1]; ++j) {
                    for (size_t k = 0; k < range[2]; ++k) {
                        std::forward<F>(func)(sycl::id<dimensions>{i, j, k});
                    }
                }
            }
        } else {
            static_assert(false, "ND not implemented yet");
        }
    }

    template<typename DataT>
    inline constexpr auto vec(const DataT& value) const {
        auto result = sycl::vec<int, dimensions>(0);

        if constexpr (dimensions > 3) {
            static_assert(false, "ND not implemented yet");
        }
        if constexpr (dimensions > 0) {
            result[0] = value[0];
        }
        if constexpr (dimensions > 1) {
            result[1] = value[1];
        }
        if constexpr (dimensions > 2) {
            result[2] = value[2];
        }
        return result;
    }
};

}  // namespace hok

#endif  // HOK_HPP
