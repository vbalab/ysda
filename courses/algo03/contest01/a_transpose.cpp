#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

// const std::string kInput = "a_tests_samples/01";
// const std::string kOutput = "a_tests_samples/01.b";
const std::string kInput = "input.bin";
const std::string kOutput = "output.bin";

constexpr std::streamoff kMatrixOffset = 8;
constexpr int32_t kBlockSize = 512;
constexpr int32_t kBlockBites = kBlockSize * kBlockSize;

void ReadHeader(std::ifstream& fin, int32_t& nrows, int32_t& ncols) {
    char header[kMatrixOffset];
    fin.seekg(0, std::ios::beg);
    fin.read(header, kMatrixOffset);
    std::memcpy(&nrows, header + 0, 4);
    std::memcpy(&ncols, header + 4, 4);
}

void WriteHeader(std::ofstream& fout, int32_t nrows, int32_t ncols) {
    char header[kMatrixOffset];
    std::memcpy(header + 0, &ncols, 4);
    std::memcpy(header + 4, &nrows, 4);
    fout.seekp(0, std::ios::beg);
    fout.write(header, kMatrixOffset);
}

void ReadBlock /*NOLINT*/ (std::ifstream& fin, std::vector<char>& iblock,
                           int32_t rows, int32_t cols, int32_t block_row,
                           int32_t block_col, int32_t ncols) {
    std::streamoff in_start = kMatrixOffset + block_row * ncols + block_col;
    fin.seekg(in_start, std::ios::beg);

    for (int32_t row = 0; row < rows; ++row) {
        fin.read(iblock.data() + row * cols, cols);

        if (row < rows - 1) {
            fin.seekg(ncols - cols, std::ios::cur);
        }
    }
}

void WriteBlock /*NOLINT*/ (std::ofstream& fout, std::vector<char>& oblock,
                            int32_t rows, int32_t cols, int32_t block_row,
                            int32_t block_col, int32_t nrows) {
    std::streamoff out_start = kMatrixOffset + block_col * nrows + block_row;
    fout.seekp(out_start, std::ios::beg);

    for (int32_t col = 0; col < cols; ++col) {
        fout.write(oblock.data() + col * rows, rows);

        if (col < cols - 1) {
            fout.seekp(nrows - rows, std::ios::cur);
        }
    }
}

void TransposeBlock(std::vector<char>& iblock, std::vector<char>& oblock,
                    int32_t rows, int32_t cols) {
    for (int32_t row = 0; row < rows; ++row) {
        for (int32_t col = 0; col < cols; ++col) {
            oblock[col * rows + row] = iblock[row * cols + col];
        }
    }
}

void TransposeColumnShortMatrix(std::ifstream& fin, std::ofstream& fout,
                                int32_t nrows, int32_t ncols) {
    int32_t cols = ncols;
    int32_t rows_per_chunk = std::max<int32_t>(
        1, (kBlockSize * kBlockSize) / (2 * std::max(1, cols)));

    for (int32_t block_row = 0; block_row < nrows;
         block_row += rows_per_chunk) {
        int32_t rows = std::min(rows_per_chunk, nrows - block_row);

        std::vector<char> iblock(rows * cols);
        std::vector<char> oblock(rows * cols);

        std::streamoff pos =
            kMatrixOffset + static_cast<std::streamoff>(block_row) *
                                static_cast<std::streamoff>(ncols);
        fin.seekg(pos, std::ios::beg);
        fin.read(iblock.data(), static_cast<std::streamsize>(rows) * cols);

        TransposeBlock(iblock, oblock, rows, cols);

        WriteBlock(fout, oblock, rows, cols, block_row, /*block_col=*/0, nrows);
    }
}

void TransposeRowShortMatrix(std::ifstream& fin, std::ofstream& fout,
                             int32_t nrows, int32_t ncols) {
    int32_t rows = nrows;
    int32_t cols_per_chunk = std::max<int32_t>(
        1, (kBlockSize * kBlockSize) / (2 * std::max(1, rows)));

    for (int32_t block_col = 0; block_col < ncols;
         block_col += cols_per_chunk) {
        int32_t cols = std::min(cols_per_chunk, ncols - block_col);

        std::vector<char> iblock(rows * cols);
        std::vector<char> oblock(rows * cols);

        ReadBlock(fin, iblock, rows, cols, /*block_row=*/0, block_col, ncols);

        TransposeBlock(iblock, oblock, rows, cols);

        std::streamoff pos =
            kMatrixOffset + static_cast<std::streamoff>(block_col) *
                                static_cast<std::streamoff>(nrows);
        fout.seekp(pos, std::ios::beg);
        fout.write(oblock.data(), static_cast<std::streamsize>(rows) * cols);
    }
}

void TransposeMatrix(std::ifstream& fin, std::ofstream& fout, int32_t nrows,
                     int32_t ncols) {
    if (ncols < kBlockSize) {
        TransposeColumnShortMatrix(fin, fout, nrows, ncols);
        return;
    }

    if (nrows < kBlockSize) {
        TransposeRowShortMatrix(fin, fout, nrows, ncols);
        return;
    }

    for (int32_t block_row = 0; block_row < nrows; block_row += kBlockSize) {
        int32_t rows = std::min(kBlockSize, nrows - block_row);

        for (int32_t block_col = 0; block_col < ncols;
             block_col += kBlockSize) {
            int32_t cols = std::min(kBlockSize, ncols - block_col);

            std::vector<char> iblock(rows * cols);
            std::vector<char> oblock(rows * cols);

            ReadBlock(fin, iblock, rows, cols, block_row, block_col, ncols);
            TransposeBlock(iblock, oblock, rows, cols);
            WriteBlock(fout, oblock, rows, cols, block_row, block_col, nrows);
        }
    }
}

int main() {
    std::ifstream fin(kInput, std::ios::binary);
    std::ofstream fout(kOutput, std::ios::binary | std::ios::trunc);

    int32_t nrows;
    int32_t ncols;

    ReadHeader(fin, nrows, ncols);
    WriteHeader(fout, nrows, ncols);

    TransposeMatrix(fin, fout, nrows, ncols);

    fout.flush();
    return 0;
}
