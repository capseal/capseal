#include "enn/cell.hpp"
#include "enn/regularizers.hpp"
#include <iostream>
#include <Eigen/Eigenvalues>

using namespace enn;

bool test_psd_constraint() {
    const int k = 8;
    const int input_dim = 3;
    const int hidden_dim = 16;
    
    EntangledCell cell(k, input_dim, hidden_dim, 0.1, 123);
    
    // Test that E = L * L^T is always PSD
    Mat E = cell.get_entanglement_matrix();
    
    Eigen::SelfAdjointEigenSolver<Mat> solver(E);
    if (solver.info() != Eigen::Success) {
        std::cout << "FAIL: Could not compute eigenvalues of E" << std::endl;
        return false;
    }
    
    Vec eigenvals = solver.eigenvalues();
    F min_eigenval = eigenvals.minCoeff();
    
    if (min_eigenval < -1e-12) {
        std::cout << "FAIL: Entanglement matrix is not PSD, min eigenvalue: " 
                  << min_eigenval << std::endl;
        return false;
    }
    
    // Test the built-in PSD check
    if (!cell.is_entanglement_psd()) {
        std::cout << "FAIL: Built-in PSD check failed but eigenvalues are non-negative" << std::endl;
        return false;
    }
    
    std::cout << "PSD constraint satisfied, min eigenvalue: " << min_eigenval << std::endl;
    return true;
}

bool test_spectral_clip() {
    // Create a matrix with some large eigenvalues
    Mat A(4, 4);
    A << 1, 0, 0, 0,
         0, 5, 0, 0,
         0, 0, 10, 0,
         0, 0, 0, 15;
    
    F max_allowed = 8.0;
    Mat A_clipped = A;
    spectral_clip(A_clipped, max_allowed);
    
    // Check that all eigenvalues are now <= max_allowed
    Eigen::SelfAdjointEigenSolver<Mat> solver(A_clipped);
    if (solver.info() != Eigen::Success) {
        std::cout << "FAIL: Could not compute eigenvalues after clipping" << std::endl;
        return false;
    }
    
    Vec eigenvals = solver.eigenvalues();
    F max_eigenval = eigenvals.maxCoeff();
    
    if (max_eigenval > max_allowed + 1e-10) {
        std::cout << "FAIL: Spectral clipping failed, max eigenvalue: " 
                  << max_eigenval << " > " << max_allowed << std::endl;
        return false;
    }
    
    // Check that small eigenvalues are clipped to 0
    F min_eigenval = eigenvals.minCoeff();
    if (min_eigenval < -1e-12) {
        std::cout << "FAIL: Negative eigenvalue after clipping: " << min_eigenval << std::endl;
        return false;
    }
    
    std::cout << "Spectral clipping successful, eigenvalues in range [" 
              << min_eigenval << ", " << max_eigenval << "]" << std::endl;
    return true;
}

bool test_symmetry_penalty() {
    // Create an asymmetric matrix
    Mat A(3, 3);
    A << 1, 2, 3,
         4, 5, 6,
         7, 8, 9;
    
    F penalty = sym_penalty(A);
    
    // Penalty should be > 0 for asymmetric matrix
    if (penalty <= 0) {
        std::cout << "FAIL: Symmetry penalty is not positive for asymmetric matrix: " 
                  << penalty << std::endl;
        return false;
    }
    
    // Penalty should be 0 for symmetric matrix
    Mat B = (A + A.transpose()) / 2.0;  // Make symmetric
    F penalty_sym = sym_penalty(B);
    
    if (penalty_sym > 1e-12) {
        std::cout << "FAIL: Symmetry penalty is not zero for symmetric matrix: " 
                  << penalty_sym << std::endl;
        return false;
    }
    
    std::cout << "Symmetry penalty test passed, asymmetric penalty: " << penalty 
              << ", symmetric penalty: " << penalty_sym << std::endl;
    return true;
}

int main() {
    std::cout << "Testing PSD constraint..." << std::endl;
    if (!test_psd_constraint()) {
        return 1;
    }
    std::cout << "PASS: PSD constraint test" << std::endl;
    
    std::cout << "Testing spectral clipping..." << std::endl;
    if (!test_spectral_clip()) {
        return 1;
    }
    std::cout << "PASS: Spectral clipping test" << std::endl;
    
    std::cout << "Testing symmetry penalty..." << std::endl;
    if (!test_symmetry_penalty()) {
        return 1;
    }
    std::cout << "PASS: Symmetry penalty test" << std::endl;
    
    std::cout << "All PSD tests passed!" << std::endl;
    return 0;
}