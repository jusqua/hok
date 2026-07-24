/* hok - v0.0.0 - Public Domain - https://github.com/jusqua/hok */

#pragma once

#include <sycl/sycl.hpp>

namespace hok::detail {

template<int dimensions>
inline constexpr auto get_linear_id(const sycl::range<dimensions>& extent, const sycl::id<dimensions>& index) {
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

template<int dimensions>
inline constexpr auto get_linear_id(const sycl::item<dimensions>& item, const sycl::vec<int, dimensions>& displacement) {
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

template<int dimensions>
inline constexpr auto get_linear_id(const sycl::item<dimensions>& item) {
    return get_linear_id(item.get_range(), item.get_id());
}

inline constexpr auto read(const float* data, size_t index) {
    auto value = sycl::float4{0.0f};
    for(int i = 0; i < 4; ++i) {
        value[i] = data[index * 4 + i];
    }
    return value;
}

template<int dimensions>
inline constexpr auto read(const float* data, const sycl::item<dimensions>& item) {
    return read(data, item.get_linear_id());
}

inline constexpr auto write(float* data, size_t index, const sycl::float4& value) {
    for(int i = 0; i < 4; ++i) {
        data[index * 4 + i] = value[i];
    }
}

template<int dimensions>
inline constexpr auto write(float* data, const sycl::item<dimensions>& item, const sycl::float4& value) {
    write(data, item.get_linear_id(), value);
}

template<typename DataT, int dimensions>
inline constexpr auto vec(const DataT& value) {
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

template <typename F, int dimensions>
inline constexpr auto map(const sycl::range<dimensions>& range, const F&& apply) {
    if constexpr (dimensions == 1) {
        for (size_t i = 0; i < range[0]; ++i) {
            apply(sycl::id<dimensions>{i});
        }
    } else if constexpr (dimensions == 2) {
        for (size_t i = 0; i < range[0]; ++i) {
            for (size_t j = 0; j < range[1]; ++j) {
                apply(sycl::id<dimensions>{i, j});
            }
        }
    } else if constexpr (dimensions == 3) {
        for (size_t i = 0; i < range[0]; ++i) {
            for (size_t j = 0; j < range[1]; ++j) {
                for (size_t k = 0; k < range[2]; ++k) {
                    apply(sycl::id<dimensions>{i, j, k});
                }
            }
        }
    } else {
        static_assert(false, "ND not implemented yet");
    }
}

} // namespace hok::detail

namespace hok::kernel {

template <int dimensions>
class invert {
public:
    invert(const sycl::range<dimensions>& data_extent, const float* input_data, float* output_data)
        : m_data_extent(data_extent), m_input_data(input_data), m_output_data(output_data) {}

    void operator()(const sycl::item<dimensions> item) const {
        detail::write(m_output_data, item, 1.0f - detail::read(m_input_data, item));
    }

private:
    const sycl::range<dimensions> m_data_extent;
    const float* m_input_data;
    float* m_output_data;
};

template <int dimensions>
class thresh {
public:
    thresh(const sycl::range<dimensions>& data_extent, const float* input_data, float* output_data, float threshold, float max_value = 1.0f, float min_value = 0.0f)
        : m_data_extent(data_extent), m_input_data(input_data), m_output_data(output_data), m_threshold(threshold), m_max_value(max_value), m_min_value(min_value) {}

    void operator()(const sycl::item<dimensions> item) const {
        auto px = detail::read(m_input_data, item);
        px.x() = px.x() > m_threshold ? m_max_value : m_min_value;
        px.y() = px.y() > m_threshold ? m_max_value : m_min_value;
        px.z() = px.z() > m_threshold ? m_max_value : m_min_value;
        detail::write(m_output_data, item, px);
    }

private:
    const sycl::range<dimensions> m_data_extent;
    const float* m_input_data;
    float* m_output_data;
    float m_threshold;
    float m_max_value;
    float m_min_value;
};

template <int dimensions>
class gray {
public:
    gray(const sycl::range<dimensions>& data_extent, const float* input_data, float* output_data)
        : m_data_extent(data_extent), m_input_data(input_data), m_output_data(output_data) {}

    void operator()(const sycl::item<dimensions> item) const {
        auto px = detail::read(m_input_data, item);
        float gray = px.x() * 0.299f + px.y() * 0.587f + px.z() * 0.114f;
        px.x() = px.y() = px.z() = gray;

        detail::write(m_output_data, item, px);
    }

private:
    const sycl::range<dimensions> m_data_extent;
    const float* m_input_data;
    float* m_output_data;
};

template <int dimensions>
class binary {
public:
    binary(const sycl::range<dimensions>& data_extent, const float* input_data, float* output_data,
        float threshold, float max_value = 1.0f, float min_value = 0.0f)
        : m_gray(data_extent, input_data, output_data),
          m_thresh(data_extent, output_data, output_data, threshold, max_value, min_value) {}

    void operator()(const sycl::item<dimensions> item) const {
        m_gray(item);
        m_thresh(item);
    }

private:
    gray<dimensions> m_gray;
    thresh<dimensions> m_thresh;
};

template <int dimensions>
class min {
public:
    min(const sycl::range<dimensions>& data_extent, const float* input1_data, const float* input2_data, float* output_data)
        : m_data_extent(data_extent), m_input1_data(input1_data), m_input2_data(input2_data), m_output_data(output_data) {}

    void operator()(const sycl::item<dimensions> item) const {
        detail::write(m_output_data, item, sycl::min(detail::read(m_input1_data, item), detail::read(m_input2_data, item)));
    }

private:
    const sycl::range<dimensions> m_data_extent;
    const float* m_input1_data;
    const float* m_input2_data;
    float* m_output_data;
};

template <int dimensions>
class max {
public:
    max(const sycl::range<dimensions>& data_extent, const float* input1_data, const float* input2_data, float* output_data)
        : m_data_extent(data_extent), m_input1_data(input1_data), m_input2_data(input2_data), m_output_data(output_data) {}

    void operator()(const sycl::item<dimensions> item) const {
        detail::write(m_output_data, item, sycl::max(detail::read(m_input1_data, item), detail::read(m_input2_data, item)));
    }

private:
    const sycl::range<dimensions> m_data_extent;
    const float* m_input1_data;
    const float* m_input2_data;
    float* m_output_data;
};

template <int dimensions>
class sum {
public:
    sum(const sycl::range<dimensions>& data_extent, const float* input1_data, const float* input2_data, float* output_data)
        : m_data_extent(data_extent), m_input1_data(input1_data), m_input2_data(input2_data), m_output_data(output_data) {}

    void operator()(const sycl::item<dimensions> item) const {
        detail::write(m_output_data, item, sycl::min(sycl::float4(1.0f), detail::read(m_input1_data, item) + detail::read(m_input2_data, item)));
    }

private:
    const sycl::range<dimensions> m_data_extent;
    const float* m_input1_data;
    const float* m_input2_data;
    float* m_output_data;
};

template <int dimensions>
class sub {
public:
    sub(const sycl::range<dimensions>& data_extent, const float* input1_data, const float* input2_data, float* output_data)
        : m_data_extent(data_extent), m_input1_data(input1_data), m_input2_data(input2_data), m_output_data(output_data) {}

    void operator()(const sycl::item<dimensions> item) const {
        detail::write(m_output_data, item, sycl::max(sycl::float4(0.0f), detail::read(m_input1_data, item) + detail::read(m_input2_data, item)));
    }

private:
    const sycl::range<dimensions> m_data_extent;
    const float* m_input1_data;
    const float* m_input2_data;
    float* m_output_data;
};

template <int dimensions>
class convolve {
public:
    convolve(const sycl::range<dimensions>& data_extent, const float* input_data, float* output_data, const sycl::range<dimensions>& window_extent, const float* window_data)
        : m_data_extent(data_extent), m_input_data(input_data), m_output_data(output_data),
          m_window_extent(window_extent), m_window_halo(detail::vec(window_extent) / 2), m_window_data(window_data) {}

    void operator()(const sycl::item<dimensions> item) const {
        auto result = sycl::float4{0};
        detail::map(m_window_extent, [&](sycl::id<dimensions> fid) {
            auto px = detail::read(m_input_data, detail::get_linear_id(item, detail::vec(fid) - m_window_halo));
            auto value = m_window_data[detail::get_linear_id(m_window_extent, fid)];
            result += px * value;
        });
        detail::write(m_output_data, item, result);
    }

private:
    const sycl::range<dimensions> m_data_extent;
    const sycl::range<dimensions> m_window_extent;
    const sycl::vec<int, dimensions> m_window_halo;
    const float* m_input_data;
    const float* m_window_data;
    float* m_output_data;
};

template <int dimensions>
class erode {
public:
    erode(const sycl::range<dimensions>& data_extent, const float* input_data, float* output_data, const sycl::range<dimensions>& window_extent, const float* window_data)
        : m_data_extent(data_extent), m_input_data(input_data), m_output_data(output_data),
          m_window_extent(window_extent), m_window_halo(detail::vec(window_extent) / 2), m_window_data(window_data) {}

    void operator()(const sycl::item<dimensions> item) const {
        auto result = sycl::float4(1.0f, 1.0f, 1.0f, 1.0f);
        auto result_sum = result.x() + result.y() + result.z();

        detail::map(m_window_extent, [&](sycl::id<dimensions> fid) {
            auto px = detail::read(m_input_data, detail::get_linear_id(item, detail::vec(fid) - m_window_halo));
            auto value = m_window_data[detail::get_linear_id(m_window_extent, fid)];
            auto px_sum = px.x() + px.y() + px.z();
            if (value != 0.0f && result_sum > px_sum) {
                result = px;
                result_sum = px_sum;
            }
        });
        detail::write(m_output_data, item, result);
    }

private:
    const sycl::range<dimensions> m_data_extent;
    const sycl::range<dimensions> m_window_extent;
    const sycl::vec<int, dimensions> m_window_halo;
    const float* m_input_data;
    const float* m_window_data;
    float* m_output_data;
};

template <int dimensions>
class dilate {
public:
    dilate(const sycl::range<dimensions>& data_extent, const float* input_data, float* output_data, const sycl::range<dimensions>& window_extent, const float* window_data)
        : m_data_extent(data_extent), m_input_data(input_data), m_output_data(output_data),
          m_window_extent(window_extent), m_window_halo(detail::vec(window_extent) / 2), m_window_data(window_data) {}

    void operator()(const sycl::item<dimensions> item) const {
        auto result = sycl::float4(0.0f, 0.0f, 0.0f, 0.0f);
        auto result_sum = result.x() + result.y() + result.z();

        detail::map(m_window_extent, [&](sycl::id<dimensions> fid) {
            auto px = detail::read(m_input_data, detail::get_linear_id(item, detail::vec(fid) - m_window_halo));
            auto value = m_window_data[detail::get_linear_id(m_window_extent, fid)];
            auto px_sum = px.x() + px.y() + px.z();
            if (value != 0.0f && result_sum < px_sum) {
                result = px;
                result_sum = px_sum;
            }
        });
        detail::write(m_output_data, item, result);
    }

private:
    const sycl::range<dimensions> m_data_extent;
    const sycl::range<dimensions> m_window_extent;
    const sycl::vec<int, dimensions> m_window_halo;
    const float* m_input_data;
    const float* m_window_data;
    float* m_output_data;
};

template <int dimensions>
class geodesic_erode {
public:
    geodesic_erode(
        const sycl::range<dimensions>& data_extent, const float* marker_data, const float* mask_data, float* output_data,
        const sycl::range<dimensions>& window_extent, const float* window_data)
        : m_erode(data_extent, marker_data, output_data, window_extent, window_data),
          m_max(data_extent, mask_data, output_data, output_data) {}

    void operator()(const sycl::item<dimensions> item) const {
        m_erode(item);
        m_max(item);
    }

private:
    erode<dimensions> m_erode;
    max<dimensions> m_max;
};

template <int dimensions>
class geodesic_dilate {
public:
    geodesic_dilate(
        const sycl::range<dimensions>& data_extent, const float* marker_data, const float* mask_data, float* output_data,
        const sycl::range<dimensions>& window_extent, const float* window_data)
        : m_dilate(data_extent, marker_data, output_data, window_extent, window_data),
          m_min(data_extent, mask_data, output_data, output_data) {}

    void operator()(const sycl::item<dimensions> item) const {
        m_dilate(item);
        m_min(item);
    }

private:
    dilate<dimensions> m_dilate;
    min<dimensions> m_min;
};

}  // namespace hok::kernel

namespace hok {

template <int dimensions>
inline sycl::event gray(
    sycl::queue& queue,
    const sycl::range<dimensions>& data_extent, const float* input_data, float* output_data) {
    return queue.parallel_for(data_extent, kernel::gray(data_extent, input_data, output_data));
}

template <int dimensions>
inline sycl::event thresh(
    sycl::queue& queue,
    const sycl::range<dimensions>& data_extent, const float* input_data, float* output_data,
    float threshold, float max_value = 1.0f, float min_value = 0.0f) {
    return queue.parallel_for(data_extent, kernel::thresh(data_extent, input_data, output_data, threshold, max_value, min_value));
}

template <int dimensions>
inline sycl::event binary(
    sycl::queue& queue,
    const sycl::range<dimensions>& data_extent, const float* input_data, float* output_data,
    float threshold, float max_value = 1.0f, float min_value = 0.0f) {
    return queue.parallel_for(data_extent, kernel::binary(data_extent, input_data, output_data, threshold, max_value, min_value));
}

template <int dimensions>
inline sycl::event min(
    sycl::queue& queue,
    const sycl::range<dimensions>& data_extent, const float* input1_data, const float* input2_data, float* output_data) {
    return queue.parallel_for(data_extent, kernel::min(data_extent, input1_data, input2_data, output_data));
}

template <int dimensions>
inline sycl::event max(
    sycl::queue& queue,
    const sycl::range<dimensions>& data_extent, const float* input1_data, const float* input2_data, float* output_data) {
    return queue.parallel_for(data_extent, kernel::max(data_extent, input1_data, input2_data, output_data));
}

template <int dimensions>
inline sycl::event sum(
    sycl::queue& queue,
    const sycl::range<dimensions>& data_extent, const float* input1_data, const float* input2_data, float* output_data) {
    return queue.parallel_for(data_extent, kernel::sum(data_extent, input1_data, input2_data, output_data));
}

template <int dimensions>
inline sycl::event sub(
    sycl::queue& queue,
    const sycl::range<dimensions>& data_extent, const float* input1_data, const float* input2_data, float* output_data) {
    return queue.parallel_for(data_extent, kernel::sub(data_extent, input1_data, input2_data, output_data));
}

template <int dimensions>
inline sycl::event convolve(
    sycl::queue& queue,
    const sycl::range<dimensions>& data_extent, const float* input_data, float* output_data,
    const sycl::range<dimensions>& window_extent, const float* window_data) {
    return queue.parallel_for(data_extent, kernel::convolve(data_extent, input_data, output_data, window_extent, window_data));
}

template <int dimensions>
inline sycl::event erode(
    sycl::queue& queue,
    const sycl::range<dimensions>& data_extent, const float* input_data, float* output_data,
    const sycl::range<dimensions>& window_extent, const float* window_data) {
    return queue.parallel_for(data_extent, kernel::erode(data_extent, input_data, output_data, window_extent, window_data));
}

template <int dimensions>
inline sycl::event dilate(
    sycl::queue& queue,
    const sycl::range<dimensions>& data_extent, const float* input_data, float* output_data,
    const sycl::range<dimensions>& window_extent, const float* window_data) {
    return queue.parallel_for(data_extent, kernel::dilate(data_extent, input_data, output_data, window_extent, window_data));
}

template <int dimensions>
inline sycl::event geodesic_erode(
    sycl::queue& queue,
    const sycl::range<dimensions>& data_extent, const float* marker_data, const float* mask_data, float* output_data,
    const sycl::range<dimensions>& window_extent, const float* window_data) {
    return queue.parallel_for(data_extent, kernel::geodesic_erode(data_extent, marker_data, mask_data, output_data, window_extent, window_data));
}

template <int dimensions>
inline sycl::event geodesic_dilate(
    sycl::queue& queue,
    const sycl::range<dimensions>& data_extent, const float* marker_data, const float* mask_data, float* output_data,
    const sycl::range<dimensions>& window_extent, const float* window_data) {
    return queue.parallel_for(data_extent, kernel::geodesic_dilate(data_extent, marker_data, mask_data, output_data, window_extent, window_data));
}

template <int dimensions>
inline sycl::event open(
    sycl::queue& queue,
    const sycl::range<dimensions>& data_extent, const float* input_data, float* output_data, float* buffer_data,
    const sycl::range<dimensions>& window_extent, const float* window_data) {
    return queue.parallel_for(data_extent,
        kernel::erode(data_extent, input_data, buffer_data, window_extent, window_data),
        kernel::dilate(data_extent, buffer_data, output_data, window_extent, window_data));
}

template <int dimensions>
inline sycl::event close(
    sycl::queue& queue,
    const sycl::range<dimensions>& data_extent, const float* input_data, float* output_data, float* buffer_data,
    const sycl::range<dimensions>& window_extent, const float* window_data) {
    return queue.parallel_for(data_extent,
        kernel::dilate(data_extent, input_data, buffer_data, window_extent, window_data),
        kernel::erode(data_extent, buffer_data, output_data, window_extent, window_data));
}

template <int dimensions>
inline sycl::event tophat(
    sycl::queue& queue,
    const sycl::range<dimensions>& data_extent, const float* input_data, float* output_data, float* buffer_data,
    const sycl::range<dimensions>& window_extent, const float* window_data) {
    return queue.parallel_for(data_extent,
        open(queue, data_extent, input_data, buffer_data, output_data, window_extent, window_data),
        kernel::sub(data_extent, input_data, buffer_data, output_data));
}

}  // namespace hok

/*
    ------------------------------------------------------------------------------
    This software is available under 2 licenses -- choose whichever you prefer.
    ------------------------------------------------------------------------------
    ALTERNATIVE A - MIT License
    Copyright (c) 2026 Ádrian Gama
    Permission is hereby granted, free of charge, to any person obtaining a copy of
    this software and associated documentation files (the "Software"), to deal in
    the Software without restriction, including without limitation the rights to
    use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
    of the Software, and to permit persons to whom the Software is furnished to do
    so, subject to the following conditions:
    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.
    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.
    ------------------------------------------------------------------------------
    ALTERNATIVE B - Public Domain (www.unlicense.org)
    This is free and unencumbered software released into the public domain.
    Anyone is free to copy, modify, publish, use, compile, sell, or distribute this
    software, either in source code form or as a compiled binary, for any purpose,
    commercial or non-commercial, and by any means.
    In jurisdictions that recognize copyright laws, the author or authors of this
    software dedicate any and all copyright interest in the software to the public
    domain. We make this dedication for the benefit of the public at large and to
    the detriment of our heirs and successors. We intend this dedication to be an
    overt act of relinquishment in perpetuity of all present and future rights to
    this software under copyright law.
    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
    ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
    WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
    ------------------------------------------------------------------------------
*/
