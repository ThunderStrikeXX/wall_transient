#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <omp.h>
#include <array>
#include <algorithm>
#include <cstddef>

// =======================================================================
//
//                       [MATERIAL PROPERTIES]
//
// =======================================================================

#pragma region steel_properties

/**
 * @brief Provides material properties for a specific type of steel.
 *
 * This namespace contains constant lookup tables and helper functions to retrieve
 * temperature-dependent thermodynamic properties of steel, specifically:
 * - Specific Heat Capacity (Cp)
 * - Density (rho)
 * - Thermal Conductivity (k)
 *
 * All functions accept temperature in **Kelvin [K]** and return values in
 * standard SI units.
 */
namespace steel {

    // Temperature [K]
    constexpr std::array<double, 15> T = { 300,400,500,600,700,800,900,1000,1100,1200,1300,1400,1500,1600,1700 };

    // Specific heat [J kg^-1 K^-1]
    constexpr std::array<double, 15> Cp_J_kgK = { 510.0296,523.4184,536.8072,550.1960,564.0032,577.3920,590.7808,604.1696,617.5584,631.3656,644.7544,658.1432,671.5320,685.3392,698.7280 };

    // Specific heat interpolation in temperature with complexity O(1)
    inline double cp(double Tquery) {

        if (Tquery <= T.front()) return Cp_J_kgK.front();
        if (Tquery >= T.back())  return Cp_J_kgK.back();

        int i = static_cast<int>((Tquery - 300.0) / 100.0);

        if (i < 0) i = 0;

        int iMax = static_cast<int>(T.size()) - 2;

        if (i > iMax) i = iMax;

        double x0 = 300.0 + 100.0 * i, x1 = x0 + 100.0;
        double y0 = Cp_J_kgK[static_cast<std::size_t>(i)];
        double y1 = Cp_J_kgK[static_cast<std::size_t>(i + 1)];
        double t = (Tquery - x0) / (x1 - x0);

        return y0 + t * (y1 - y0);
    }

    // Density [kg/m^3]
    double rho(double T) { return (7.9841 - 2.6560e-4 * T - 1.158e-7 * T * T) * 1e3; }

    // Thermal conductivity [W/(m·K)]
    double k(double T) { return (3.116e-2 + 1.618e-4 * T) * 100.0; }
}

#pragma endregion

// =======================================================================
//
//                        [VARIOUS ALGORITHMS]
//
// =======================================================================

#pragma region various_algorithms

/**
 * @brief Solves a tridiagonal system of linear equations A*x = d using the Thomas Algorithm (TDMA).
 *
 * This function efficiently solves a system where the coefficient matrix A is tridiagonal,
 * meaning it only has non-zero elements on the main diagonal, the sub-diagonal, and the super-diagonal.
 * The system is defined by:
 * - 'a': The sub-diagonal (below the main diagonal). a[0] is typically unused.
 * - 'b': The main diagonal.
 * - 'c': The super-diagonal (above the main diagonal). c[N-1] is typically unused.
 * - 'd': The right-hand side vector.
 *
 * The method consists of two main phases: forward elimination and back substitution,
 * which is optimized for the sparse tridiagonal structure.
 *
 * @param a The sub-diagonal vector (size N, with a[0] often being zero/unused).
 * @param b The main diagonal vector (size N). Must contain non-zero elements.
 * @param c The super-diagonal vector (size N, with c[N-1] often being zero/unused).
 * @param d The right-hand side vector (size N).
 * @return std::vector<double> The solution vector 'x' (size N).
 * * @note This implementation assumes the system is diagonally dominant or otherwise
 * stable, as it does not include pivoting. The vectors 'a', 'b', 'c', and 'd' must
 * all have the same size N, corresponding to the size of the system.
 */
std::vector<double> solveTridiagonal(const std::vector<double>& a,
    const std::vector<double>& b,
    const std::vector<double>& c,
    const std::vector<double>& d) {
    int n = b.size();
    std::vector<double> c_star(n, 0.0);
    std::vector<double> d_star(n, 0.0);
    std::vector<double> x(n, 0.0);

    c_star[0] = c[0] / b[0];
    d_star[0] = d[0] / b[0];

    for (int i = 1; i < n; i++) {
        double m = b[i] - a[i] * c_star[i - 1];
        c_star[i] = c[i] / m;
        d_star[i] = (d[i] - a[i] * d_star[i - 1]) / m;
    }

    x[n - 1] = d_star[n - 1];
    for (int i = n - 2; i >= 0; i--) {
        x[i] = d_star[i] - c_star[i] * x[i + 1];
    }

    return x;
}

#pragma endregion

//-------------------------------------------------------------
// Simplified 1D transient coupling between wall, wick, and vapor
//-------------------------------------------------------------

int main() {
    const double L = 1.0;                                   // Length of the heat pipe [m]
    const int Nx = 10000;                                   // Number of nodes [-]
    const double dx = L / Nx;                               // Spacing between nodes [-]
    const int t_max = 20;                                   // Maximum simulation time [s]
    const double dt = 0.001;                                     // Initial timestep [s] 
    const double Nt = (int)std::round(t_max / dt);          // Number of iterations [-]
	const double T_0 = 500.0;                               // Initial temperature [K]

    // Temperature scalar field
    std::vector<std::vector<double>> T(Nx, std::vector<double>(Nt, T_0));

    std::vector<double> a(Nx, 0.0);
    std::vector<double> b(Nx, 0.0);
    std::vector<double> c(Nx, 0.0);
    std::vector<double> d(Nx, 0.0);

    // Node 0: imposed temperature T = T_0
    b[0] = 1.0;  c[0] = 0.0;  a[0] = 0.0; d[0] = T_0;

    // Nodo Nx - 1: zero heat flux
    b[Nx - 1] = 1.0;  a[Nx - 1] = -1.0;  c[Nx - 1] = 0.0; d[Nx - 1] = 0.0;

	double Cp_node, k_node, rho_node;

	// Source term
    std::vector<double> Q(Nx, 1e8);

	// Time-stepping loop
    for (int it = 1; it < Nt; it++) {        

		std::cout << "Time step " << it << " / " << Nt - 1 << "\n";

		// d coefficients assembly loop
        #pragma omp parallel
        for (int ix = 1; ix < Nx - 1; ix++) {

            Cp_node = steel::cp(T[ix][it - 1]);
			rho_node = steel::rho(T[ix][it - 1]);
			k_node = steel::k(T[ix][it - 1]);

            double alpha = k_node / (rho_node * Cp_node);
            double r = alpha * dt / (dx * dx);

            // Coefficients
            a[ix] = -r;
            b[ix] = 1 + 2 * r;
            c[ix] = -r;
            d[ix] = T[ix][it - 1] + dt * Q[ix] / (rho_node * Cp_node);
        }

        std::vector<double> temp_x(Nx, 0.0);
        temp_x = solveTridiagonal(a, b, c, d);

        #pragma omp parallel for
        for (int jx = 0; jx < Nx; jx++) {
            T[jx][it] = temp_x[jx];
        }
    }

    std::ofstream fout("solution_wall_FD.txt");
    fout << std::setprecision(8);

    // Output section

    for (int ix = 0; ix < Nx; ix++) {

        if (ix == (Nx / 2)) {
            for (int it = 0; it < Nt; ++it) fout << T[ix][it] << ", ";
        }
    }

    fout.close();

    return 0;
}
