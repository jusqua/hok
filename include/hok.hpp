#ifndef HOK_HPP
#define HOK_HPP

#include <sycl/sycl.hpp>

// TODO: Support nd_range

namespace hok {

template <int dimensions>
class kernel_impl {
public:
    /// Must be implemented by derived classes
    void operator()(const sycl::item<dimensions> item) const;

protected:
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

    inline constexpr auto read(const float* data, size_t index) const {
        auto value = sycl::float4{0.0f};
        value.load(index, data);
        return value;
    }

    inline constexpr auto read(const float* data, const sycl::item<dimensions>& item) const {
        return read(data, item.get_linear_id());
    }

    inline constexpr auto write(float* data, size_t index, const sycl::float4& value) const {
        value.store(index, data);
    }

    inline constexpr auto write(float* data, const sycl::item<dimensions>& item, const sycl::float4& value) const {
        write(data, item.get_linear_id(), value);
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

    template <typename F>
    inline constexpr auto map(const sycl::range<dimensions>& range, const F&& apply) const {
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
};

template <int dimensions>
class invert : public kernel_impl<dimensions> {
public:
    invert(const sycl::range<dimensions>& data_extent, const float* input_data, float* output_data)
        : m_data_extent(data_extent), m_input_data(input_data), m_output_data(output_data) {}

    void operator()(const sycl::item<dimensions> item) const {
        this->write(m_output_data, item, 1.0f - this->read(m_input_data, item));
    }

private:
    const sycl::range<dimensions> m_data_extent;
    const float* m_input_data;
    float* m_output_data;
};

template <int dimensions>
class thresh : public kernel_impl<dimensions> {
public:
    thresh(const sycl::range<dimensions>& data_extent, const float* input_data, float* output_data, float threshold, float max_value = 1.0f, float min_value = 0.0f)
        : m_data_extent(data_extent), m_input_data(input_data), m_output_data(output_data), m_threshold(threshold), m_max_value(max_value), m_min_value(min_value) {}

    void operator()(const sycl::item<dimensions> item) const {
        auto px = this->read(m_input_data, item);
        px.x() = px.x() > m_threshold ? m_max_value : m_min_value;
        px.y() = px.y() > m_threshold ? m_max_value : m_min_value;
        px.z() = px.z() > m_threshold ? m_max_value : m_min_value;
        this->write(m_output_data, item, px);
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
class gray : public kernel_impl<dimensions> {
public:
    gray(const sycl::range<dimensions>& data_extent, const float* input_data, float* output_data)
        : m_data_extent(data_extent), m_input_data(input_data), m_output_data(output_data) {}

    void operator()(const sycl::item<dimensions> item) const {
        auto px = this->read(m_input_data, item);
        float gray = px.x() * 0.299f + px.y() * 0.587f + px.z() * 0.114f;
        px.x() = px.y() = px.z() = gray;

        this->write(m_output_data, item, px);
    }

private:
    const sycl::range<dimensions> m_data_extent;
    const float* m_input_data;
    float* m_output_data;
};

template <int dimensions>
class min : public kernel_impl<dimensions> {
public:
    min(const sycl::range<dimensions>& data_extent, const float* input1_data, const float* input2_data, float* output_data)
        : m_data_extent(data_extent), m_input1_data(input1_data), m_input2_data(input2_data), m_output_data(output_data) {}

    void operator()(const sycl::item<dimensions> item) const {
        this->write(m_output_data, item, sycl::min(this->read(m_input1_data, item), this->read(m_input2_data, item)));
    }

private:
    const sycl::range<dimensions> m_data_extent;
    const float* m_input1_data;
    const float* m_input2_data;
    float* m_output_data;
};

template <int dimensions>
class max : public kernel_impl<dimensions> {
public:
    max(const sycl::range<dimensions>& data_extent, const float* input1_data, const float* input2_data, float* output_data)
        : m_data_extent(data_extent), m_input1_data(input1_data), m_input2_data(input2_data), m_output_data(output_data) {}

    void operator()(const sycl::item<dimensions> item) const {
        this->write(m_output_data, item, sycl::max(this->read(m_input1_data, item), this->read(m_input2_data, item)));
    }

private:
    const sycl::range<dimensions> m_data_extent;
    const float* m_input1_data;
    const float* m_input2_data;
    float* m_output_data;
};

template <int dimensions>
class sum : public kernel_impl<dimensions> {
public:
    sum(const sycl::range<dimensions>& data_extent, const float* input1_data, const float* input2_data, float* output_data)
        : m_data_extent(data_extent), m_input1_data(input1_data), m_input2_data(input2_data), m_output_data(output_data) {}

    void operator()(const sycl::item<dimensions> item) const {
        this->write(m_output_data, item, sycl::min(sycl::float4(1.0f), this->read(m_input1_data, item) + this->read(m_input2_data, item)));
    }

private:
    const sycl::range<dimensions> m_data_extent;
    const float* m_input1_data;
    const float* m_input2_data;
    float* m_output_data;
};

template <int dimensions>
class convolve : public kernel_impl<dimensions> {
public:
    convolve(const sycl::range<dimensions>& data_extent, const float* input_data, float* output_data, const sycl::range<dimensions>& window_extent, const float* window_data)
        : m_data_extent(data_extent), m_input_data(input_data), m_output_data(output_data),
          m_window_extent(window_extent), m_window_halo(this->vec(window_extent) / 2), m_window_data(window_data) {}

    void operator()(const sycl::item<dimensions> item) const {
        auto result = sycl::float4{0};
        this->map(m_window_extent, [&](sycl::id<dimensions> fid) {
            auto px = this->read(m_input_data, this->get_linear_id(item, this->vec(fid) - m_window_halo));
            auto value = m_window_data[this->get_linear_id(m_window_extent, fid)];
            result += px * value;
        });
        this->write(m_output_data, item, result);
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
class erode : public kernel_impl<dimensions> {
public:
    erode(const sycl::range<dimensions>& data_extent, const float* input_data, float* output_data, const sycl::range<dimensions>& window_extent, const float* window_data)
        : m_data_extent(data_extent), m_input_data(input_data), m_output_data(output_data),
          m_window_extent(window_extent), m_window_halo(this->vec(window_extent) / 2), m_window_data(window_data) {}

    void operator()(const sycl::item<dimensions> item) const {
        auto result = sycl::float4(1.0f, 1.0f, 1.0f, 1.0f);
        auto result_sum = result.x() + result.y() + result.z();

        this->map(m_window_extent, [&](sycl::id<dimensions> fid) {
            auto px = this->read(m_input_data, this->get_linear_id(item, this->vec(fid) - m_window_halo));
            auto value = m_window_data[this->get_linear_id(m_window_extent, fid)];
            auto px_sum = px.x() + px.y() + px.z();
            if (value != 0.0f && result_sum > px_sum) {
                result = px;
                result_sum = px_sum;
            }
        });
        this->write(m_output_data, item, result);
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
class dilate : public kernel_impl<dimensions> {
public:
    dilate(const sycl::range<dimensions>& data_extent, const float* input_data, float* output_data, const sycl::range<dimensions>& window_extent, const float* window_data)
        : m_data_extent(data_extent), m_input_data(input_data), m_output_data(output_data),
          m_window_extent(window_extent), m_window_halo(this->vec(window_extent) / 2), m_window_data(window_data) {}

    void operator()(const sycl::item<dimensions> item) const {
        auto result = sycl::float4(0.0f, 0.0f, 0.0f, 0.0f);
        auto result_sum = result.x() + result.y() + result.z();

        this->map(m_window_extent, [&](sycl::id<dimensions> fid) {
            auto px = this->read(m_input_data, this->get_linear_id(item, this->vec(fid) - m_window_halo));
            auto value = m_window_data[this->get_linear_id(m_window_extent, fid)];
            auto px_sum = px.x() + px.y() + px.z();
            if (value != 0.0f && result_sum < px_sum) {
                result = px;
                result_sum = px_sum;
            }
        });
        this->write(m_output_data, item, result);
    }

private:
    const sycl::range<dimensions> m_data_extent;
    const sycl::range<dimensions> m_window_extent;
    const sycl::vec<int, dimensions> m_window_halo;
    const float* m_input_data;
    const float* m_window_data;
    float* m_output_data;
};

}  // namespace hok

#endif  // HOK_HPP
