
#ifndef ROT_MATRIX_H
#define ROT_MATRIX_H

namespace tomocam {
    template <typename T>
    class RotMatrix {
      private:
        std::array<T, 9> coeffs;

      public:
        RotMatrix(T alpha, T gamma) {
            T ca = std::cos(alpha);
            T sa = std::sin(alpha);
            T cg = std::cos(gamma);
            T sg = std::sin(gamma);

            coeffs[0] = cg;
            coeffs[1] = -sg;
            coeffs[2] = 0;
            coeffs[3] = ca * sg;
            coeffs[4] = ca * cg;
            coeffs[5] = -sa;
            coeffs[6] = sa * sg;
            coeffs[7] = sa * cg;
            coeffs[8] = ca;
        }

        std::array<T, 3> multiply(const std::array<T, 3> &vec) const {
            std::array<T, 3> result;
            for (size_t i = 0; i < 3; ++i) {
                for (size_t j = 0; j < 3; ++j) {
                    result[i] += coeffs[i * 3 + j] * vec[j];
                }
            }
            return result;
        }
        std::array<T, 3> transpose_multiply(const std::array<T, 3> &vec) const {
            std::array<T, 3> result;
            for (size_t i = 0; i < 3; ++i) {
                for (size_t j = 0; j < 3; ++j) {
                    result[i] += coeffs[j * 3 + i] * vec[j];
                }
            }
            return result;
        }
    };

} // namespace tomocam

#endif
