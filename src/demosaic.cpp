#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <vector>

#include <Eigen/Dense>
#include <Eigen/SVD>

#include "Condat_TV_1D_v2.c"
#include "tinypng.h"

constexpr size_t rows = 2048;
constexpr size_t cols = 4096;
constexpr size_t start_cols = 100;
constexpr size_t match_px_top = 350;
constexpr size_t match_px_bot = 240;

using ColVec = Eigen::Matrix<double, cols / 2, 1>;
using vecColVec = std::vector<ColVec, Eigen::aligned_allocator<ColVec>>;

constexpr size_t calib_start = 1294;
constexpr size_t calib_end = 1314;
constexpr size_t calib_n = calib_end - calib_start;

class PostProcessor {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

   private:
    Eigen::Array<double, cols / 2, -1> reds;
    Eigen::Array<double, cols / 2, -1> greens_r;
    Eigen::Array<double, cols / 2, -1> greens_b;
    Eigen::Array<double, cols / 2, -1> blues;

    size_t col_ind;
    size_t frame_ind;

    /**
     * Each column may have a slightly different exposure due to electrical
     * noise in the sensor readout. Correct for this error by performing a
     * linear regression against the *_start columns, which are averaged over
     * start_cols. The regression variables are intensity and y value (index
     * along the col).
     */
    static void matchColumnExposure(Eigen::Ref<ColVec> mutable_col,
                                    const ColVec &target) {
        constexpr size_t acols = 2;
        Eigen::Matrix<double, cols / 2, acols> A;
        A.col(0) = mutable_col;
        A.col(1).setLinSpaced(0, 1);
        A.col(1).array() *= mutable_col.array();

        Eigen::Matrix<double, match_px_bot + match_px_top, acols> AA;
        AA.block<match_px_bot, acols>(0, 0) =
            A.block<match_px_bot, acols>(0, 0);
        AA.block<match_px_top, acols>(match_px_bot, 0) =
            A.block<match_px_top, acols>(cols / 2 - match_px_top - 1, 0);

        Eigen::Matrix<double, match_px_bot + match_px_top, 1> b;
        b.segment<match_px_bot>(0) = target.segment<match_px_bot>(0);
        b.segment<match_px_top>(match_px_bot) =
            target.segment<match_px_top>(cols / 2 - match_px_top - 1);

        Eigen::Array<double, match_px_bot + match_px_top, 1> w;
        w.setOnes();
        //    1.0 / (1.0 + AA.col(0).array().sqrt());

        AA.array().colwise() *= w;
        b.array() *= w;

        Eigen::Matrix<double, acols, 1> x =
            (AA.transpose() * AA).inverse() * (AA.transpose() * b);

        mutable_col = A * x;
    }

    /**
     */
    static void matchGreens(Eigen::Ref<ColVec> mutable_col,
                            Eigen::Ref<ColVec> target) {
        // Eigen::Matrix<double, cols / 2, 3> pr;
        // pr.col(0) = mutable_col;
        // pr.col(1) = target;
        // for (int k = 0; k < 3; k++) {
        constexpr size_t acols = 7;
        Eigen::Matrix<double, cols / 2, acols> A;
        A.col(0) = mutable_col;
        A.col(1).array() = mutable_col.array().square();
        A.col(2).setLinSpaced(-1.0, 1.0);
        A.col(3).array() = A.col(2).array().square();
        A.col(2).array() *= mutable_col.array();
        A.col(3).array() *= mutable_col.array();
        A.col(4).setOnes();
        A.col(5).setLinSpaced(-1.0, 1.0);
        A.col(6).array() = A.col(5).array().square();

        ColVec b;
        b.head<cols / 2 - 1>().array() =
            0.5 * (target.tail<cols / 2 - 1>() + target.head<cols / 2 - 1>());
        b(cols / 2 - 1) = target(cols / 2 - 1);

        Eigen::Array<double, cols / 2, 1> w;
        w.setOnes();
        for (size_t i = 0; i < cols / 2; i++) {
            if (mutable_col(i) < 0.01 || target(i) < 0.01) {
                w(i) = 0;
            };
        }

        // 1.0 / (1.0 + ((A.col(0) - b).array() / 0.5).square());

        Eigen::Matrix<double, cols / 2, acols> AA = A;
        A.array().colwise() *= w;
        b.array() *= w;

        Eigen::Matrix<double, acols, 1> x =
            (A.transpose() * A).inverse() * (A.transpose() * b);

        mutable_col = AA * x;
        for (size_t i = 0; i < cols / 2; i++) {
            if (mutable_col(i) < 0) mutable_col(i) = 0;
        }
        // pr.col(2) = mutable_col;
        // std::cout << pr << std::endl;
        // std::exit(0);
        //}
    }

    static std::unique_ptr<std::vector<uint8_t>> rgbToByteArray(
        Eigen::Ref<Eigen::Array<double, cols, -1>> r,
        Eigen::Ref<Eigen::Array<double, cols, -1>> g,
        Eigen::Ref<Eigen::Array<double, cols, -1>> b) {
        auto arr = std::make_unique<std::vector<uint8_t>>(rows * cols * 2, 255);
        auto &ba = *arr;
        auto clam = [](double y) {
            if (y < 0) return (double)0;
            if (y > 1) return (double)1;
            return y;
        };

        Eigen::Array<double, cols, -1> R = r.unaryExpr(clam) * 255;
        Eigen::Array<double, cols, -1> G = b.unaryExpr(clam) * 255;
        Eigen::Array<double, cols, -1> B = b.unaryExpr(clam) * 255;

        for (size_t x = 0; x < rows / 2; x++) {
            for (size_t y = 0; y < cols; y++) {
                ba[4 * (x * cols + y)] = R(y, x);
                ba[4 * (x * cols + y) + 1] = G(y, x);
                ba[4 * (x * cols + y) + 2] = B(y, x);
            }
        }
        return arr;
    }

    static void adjustExposure(Eigen::Array<double, cols / 2, -1> &x,
                               const double m, const double k) {
        x -= x.minCoeff();
        x /= x.maxCoeff();
        // adjusts exposure with the following curve:
        // y = 1 - m * log(exp(-k*x - log(1 / (exp(k/m) - 1))) + 1) / k;
        // where m = percentile / x(percentile).
        const size_t n = x.cols();
        Eigen::Array<double, cols / 2, -1> tmp = x;
        const double x0 = std::log(1 / (std::exp(k / m) - 1.0));

        x = 1 - (m / k) * ((-k * tmp - x0).exp() + 1.0).log();
    }

   public:
    PostProcessor(size_t n)
        : reds(cols / 2, rows / 2 * n),
          greens_r(cols / 2, rows / 2 * n),
          greens_b(cols / 2, rows / 2 * n),
          blues(cols / 2, rows / 2 * n),
          col_ind(0),
          frame_ind(0) {}

    /**
     * performs demosaicing and outputs byte array
     */
    std::unique_ptr<std::vector<uint8_t>> pop() {
        size_t ind = frame_ind * rows / 2;
        Eigen::Array<double, cols, -1> greens_s(cols, rows / 2);
        Eigen::Array<double, cols / 2, -1> green_r =
            greens_r.block<cols / 2, rows / 2>(0, ind + 1);
        Eigen::Array<double, cols / 2, -1> green_b =
            greens_b.block<cols / 2, rows / 2>(0, ind + 0);

        for (size_t c = 0; c < cols; c++) {
            if (c % 2) {
                greens_s.row(c) = green_r.row(c / 2);
            } else {
                greens_s.row(c) = green_b.row(c / 2);
            }
        }
        Eigen::Array<double, cols, -1> tmp = greens_s;
        TV1D_denoise_v2(tmp.data(), greens_s.data(), cols * rows / 2, 0.02);

        Eigen::Array<double, cols, -1> reds_s(cols, rows / 2);
        Eigen::Array<double, cols, -1> blues_s(cols, rows / 2);
        for (size_t c = 0; c < cols / 2; c++) {
            reds_s.row(c * 2) = reds.block<1, rows / 2>(c, ind + 1);
            blues_s.row(c * 2 + 1) = blues.block<1, rows / 2>(c, ind + 0);
        }
        constexpr double interp_smooth = 0.1;
        for (size_t c = 0; c < cols / 2; c++) {
            if (c * 2 + 2 < cols) {
                Eigen::Array<double, 1, rows> g =
                    (greens_s.row(c * 2 + 1) + interp_smooth) /
                    (greens_s.row(c * 2) + greens_s.row(c * 2 + 2) +
                     2 * interp_smooth);
                reds_s.row(c * 2 + 1) =
                    g * (reds_s.row(c * 2) + reds_s.row(c * 2 + 2));
            } else {
                reds_s.row(c * 2 + 1) = reds_s.row(c * 2);
            }
            if (c > 0) {
                Eigen::Array<double, 1, rows> g =
                    (greens_s.row(c * 2) + interp_smooth) /
                    (greens_s.row(c * 2 - 1) + greens_s.row(c * 2 + 1) +
                     2 * interp_smooth);
                blues_s.row(c * 2) =
                    g * (blues_s.row(c * 2 - 1) + blues_s.row(c * 2 + 1));
            } else {
                blues_s.row(c * 2) = blues_s.row(c * 2 + 1);
            }
        }
        // tmp = reds_s;
        // TV1D_denoise_v2(tmp.data(), reds_s.data(), cols * rows / 2, 0.01);
        // tmp = blues_s;
        // TV1D_denoise_v2(tmp.data(), blues_s.data(), cols * rows / 2, 0.01);
        auto bytes = rgbToByteArray(reds_s, greens_s, blues_s);
        frame_ind++;
        return bytes;
    }

    void push(const std::vector<uint16_t> &raw) {
        for (size_t row = 0; row < rows / 2; row++) {
            blues.col(col_ind) =
                Eigen::Map<const Eigen::Matrix<uint16_t, cols / 2, 1>,
                           Eigen::Unaligned, Eigen::Stride<1, 2>>(
                    raw.data() + cols * row * 2 + 1)
                    .cast<double>() /
                65536;
            greens_b.col(col_ind) =
                Eigen::Map<const Eigen::Matrix<uint16_t, cols / 2, 1>,
                           Eigen::Unaligned, Eigen::Stride<1, 2>>(
                    raw.data() + cols * row * 2)
                    .cast<double>() /
                65536;
            reds.col(col_ind) =
                Eigen::Map<const Eigen::Matrix<uint16_t, cols / 2, 1>,
                           Eigen::Unaligned, Eigen::Stride<1, 2>>(
                    raw.data() + cols * (row * 2 + 1))
                    .cast<double>() /
                65536;
            greens_r.col(col_ind) =
                Eigen::Map<const Eigen::Matrix<uint16_t, cols / 2, 1>,
                           Eigen::Unaligned, Eigen::Stride<1, 2>>(
                    raw.data() + cols * (row * 2 + 1) + 1)
                    .cast<double>() /
                65536;

            if (col_ind > 0) {
                matchGreens(greens_r.col(col_ind), greens_b.col(col_ind - 1));
            }
            col_ind++;
        }
    }
    void process() {
        reds = (reds + 0.05).pow(1 / 2.20);
        greens_r = (greens_r + 0.05).pow(1 / 2.20);
        greens_b = (greens_b + 0.05).pow(1 / 2.20);
        blues = (blues + 0.05).pow(1 / 2.20);
        ColVec red_start = reds.leftCols(1000).rowwise().mean();
        ColVec green_start = (greens_r.leftCols(1000).rowwise().mean() +
                              greens_b.leftCols(1000).rowwise().mean()) *
                             0.5;
        ColVec blue_start = blues.leftCols(1000).rowwise().mean();
        const size_t n = reds.cols();
        Eigen::ArrayXd tmp0(n);
        Eigen::ArrayXd tmp1(n);

        std::cerr << "Matching column exposure" << std::endl;
        for (size_t c = 0; c < n; c++) {
            matchColumnExposure(reds.col(c), red_start);
            matchColumnExposure(greens_r.col(c), green_start);
            matchColumnExposure(greens_b.col(c), green_start);
            matchColumnExposure(blues.col(c), blue_start);

            TV1D_denoise_v2(reds.block<calib_n + 2, 1>(calib_start, c).data(),
                            tmp0.data(), calib_n + 2, 0.02);
            reds.block<calib_n + 2, 1>(calib_start, c) =
                tmp0.head<calib_n + 2>();

            TV1D_denoise_v2(
                greens_r.block<calib_n + 2, 1>(calib_start, c).data(),
                tmp0.data(), calib_n + 2, 0.02);
            greens_r.block<calib_n + 2, 1>(calib_start, c) =
                tmp0.head<calib_n + 2>();

            TV1D_denoise_v2(
                greens_b.block<calib_n + 2, 1>(calib_start, c).data(),
                tmp0.data(), calib_n + 2, 0.02);
            greens_b.block<calib_n + 2, 1>(calib_start, c) =
                tmp0.head<calib_n + 2>();

            TV1D_denoise_v2(blues.block<calib_n + 2, 1>(calib_start, c).data(),
                            tmp0.data(), calib_n + 2, 0.02);
            blues.block<calib_n + 2, 1>(calib_start, c) =
                tmp0.head<calib_n + 2>();
        }

        std::cerr << "Row denoise" << std::endl;
        for (size_t r = 0; r < cols / 2; r++) {
            tmp1 = reds.row(r);
            TV1D_denoise_v2(tmp1.data(), tmp0.data(), n, 0.007);
            reds.row(r) = tmp0;

            tmp1 = greens_r.row(r);
            TV1D_denoise_v2(tmp1.data(), tmp0.data(), n, 0.007);
            greens_r.row(r) = tmp0;

            tmp1 = greens_b.row(r);
            TV1D_denoise_v2(tmp1.data(), tmp0.data(), n, 0.007);
            greens_b.row(r) = tmp0;

            tmp1 = blues.row(r);
            TV1D_denoise_v2(tmp1.data(), tmp0.data(), n, 0.01);
            blues.row(r) = tmp0;
        }

        std::cerr << "Adjusting exposure" << std::endl;
        adjustExposure(reds, 3.8, 12);
        adjustExposure(greens_r, 4.1, 12);
        adjustExposure(greens_b, 4.1, 12);
        adjustExposure(blues, 4.5, 12);
    }
};

int main(int argc, char **argv) {
    if (argc <= 1) {
        std::cerr << "USAGE: demosaic file_1.bin file_2.bin ... file_n.bin"
                  << std::endl;
    }

    std::vector<uint16_t> buf(rows * cols);
    const int n = argc - 1;

    PostProcessor pp(n);
    for (int f = 0; f < n; f++) {
        std::string filename = argv[f + 1];
        std::ifstream fin(filename, std::ios::in | std::ios::binary);
        fin.read(reinterpret_cast<char *>(buf.data()), rows * cols * 2);
        fin.close();
        pp.push(buf);
    }
    pp.process();
    for (int f = 0; f < n - 1; f++) {
        std::string filename = argv[f + 1];
        filename[filename.size() - 1] = 'g';
        filename[filename.size() - 2] = 'n';
        filename[filename.size() - 3] = 'p';
        auto bytes = pp.pop();
        tinypng::PNG png(cols, bytes->size() / cols / 4, bytes->data());
        png.writeToFile(filename);
        std::cerr << filename << std::endl;
    }
    return 0;
}
