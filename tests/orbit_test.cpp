#include <cmath>
#include <iostream>
#include <fstream>
#include <chrono>
#include <iomanip>

#define EPIC_USE_MULTIGRID_POISSON

#include "include/types.hpp"
#include "include/kernels/math/structures/vec2.hpp"
#include "include/kernels/physics/structures/mesh.hpp"
#include "include/kernels/physics/structures/particlegroup.hpp"

#include "src/engine/core/environment.cpp"
#include "src/diagnostics/physDiag.cpp"
#include "src/io/writeBin.cpp"

using types::real;
using kernels::math::structures::vec2;
using kernels::physics::structures::_2Mesh;

using src::engine::env::_2environment;
using src::engine::env::IntegratorKind;
using src::engine::env::FieldMode;
using src::engine::env::FieldUpdateKind;

int main(){
    std::cout << "=== 2D two-particle orbit test ===\n";

    const int nx = 256;
    const int ny = 256;

    _2Mesh mesh(0.0, 1.0, 0.0, 1.0, nx, ny);

    const real dt = static_cast<real>(1e-4);
    _2environment env(mesh, dt);

    env.set_integrator(IntegratorKind::Boris);
    env.set_field_mode(FieldMode::SelfConsistent);
    env.set_field_update(FieldUpdateKind::PoissonYee);

    env.enable_gauss_projection(true);

    auto& fm = env.fields();
    auto& pm = env.particles();

    auto& Ex = fm.Ex();
    auto& Ey = fm.Ey();
    auto& Bz = fm.Bz();
    for (int j = 0; j < Ex.n_y; ++j)
        for (int i = 0; i < Ex.n_x; ++i)
            Ex.valueAt(i, j) = 0.0;
    for (int j = 0; j < Ey.n_y; ++j)
        for (int i = 0; i < Ey.n_x; ++i)
            Ey.valueAt(i, j) = 0.0;
    for (int j = 0; j < Bz.n_y; ++j)
        for (int i = 0; i < Bz.n_x; ++i)
            Bz.valueAt(i, j) = 0.0;

    const real m = static_cast<real>(1.0);
    const real q = static_cast<real>(1e-5);

    std::size_t s_idx = pm.add_species_2d(m, q);
    auto& species = pm.species_2d(s_idx);

    const real x_c  = static_cast<real>(0.5);
    const real y_c  = static_cast<real>(0.5);
    const real dx0  = static_cast<real>(0.05);

    const real v_orb = static_cast<real>(std::sqrt(0.5) * 1);

    species.add_particle(vec2{x_c - dx0, y_c}, vec2{0.0, -v_orb});
    species.add_particle(vec2{x_c + dx0, y_c}, vec2{0.0, v_orb});

    species.particles.charge[0] = -q;
    species.particles.charge[1] = q;

    fm.reset_sources();
    src::particles::deposit_rho_J_2d(mesh, species.particles,
                                     fm.rho(), fm.Jx(), fm.Jy());
    src::fields::solvers::enforce_gauss_law_yee_2d(fm);

    std::ofstream csv_file("orbit_data.csv");
    csv_file << "x1,y1,x2,y2,vx1,vy1,vx2,vy2,Fx1,Fy1,Fx2,Fy2,Px,Py,t\n";

    auto record_csv = [&](std::ofstream& csv){
        const auto& grp = species.particles;
        auto acc = kernels::physics::structures::accFromLorentzField(
            species.particles, mesh, src::coupling::make_cell_center_E(fm),
            src::coupling::make_cell_center_B(fm));
        vec2 F1 = grp.mass[0] * acc[0];
        vec2 F2 = grp.mass[1] * acc[1];
        vec2 P = src::diagnostics::total_momentum(grp);
        csv << std::fixed << std::setprecision(6)
            << grp.pos[0].x << "," << grp.pos[0].y << ","
            << grp.pos[1].x << "," << grp.pos[1].y << ","
            << grp.vel[0].x << "," << grp.vel[0].y << ","
            << grp.vel[1].x << "," << grp.vel[1].y << ","
            << F1.x << "," << F1.y << ","
            << F2.x << "," << F2.y << ","
            << P.x << "," << P.y << ","
            << env.time() << "\n";
    };

    auto dump = [&](const char* label){
        const auto& grp = species.particles;
        vec2 P = src::diagnostics::total_momentum(grp);
        auto acc = kernels::physics::structures::accFromLorentzField(
            species.particles, mesh, src::coupling::make_cell_center_E(fm),
            src::coupling::make_cell_center_B(fm));
        vec2 F1 = grp.mass[0] * acc[0];
        vec2 F2 = grp.mass[1] * acc[1];
        std::cout << std::fixed << std::setprecision(4)
                  << label
                  << " t=" << env.time()
                  << " x1=" << grp.pos[0].x << " y1=" << grp.pos[0].y
                  << " x2=" << grp.pos[1].x << " y2=" << grp.pos[1].y
                  << " vx1=" << grp.vel[0].x << " vy1=" << grp.vel[0].y
                  << " vx2=" << grp.vel[1].x << " vy2=" << grp.vel[1].y
                  << " Fx1=" << F1.x << " Fy1=" << F1.y
                  << " Fx2=" << F2.x << " Fy2=" << F2.y
                  << " Px=" << P.x << " Py=" << P.y
                  << "\n";
    };

    record_csv(csv_file);
    dump("Initial");
    src::io::write_particles_2d("orbit_particles_0.bin",
                                 species.particles, env.time());

    const int nSteps = 1500;

    const int printEvery = 100;
    auto t_start = std::chrono::steady_clock::now();

    for (int n = 0; n < nSteps; ++n){
        env.step();
        record_csv(csv_file);

        if (((n + 1) % printEvery) == 0 || (n + 1) == nSteps){
            auto t_now = std::chrono::steady_clock::now();
            std::chrono::duration<double> elapsed = t_now - t_start;
            double elapsed_s   = elapsed.count();
            double per_step    = elapsed_s / static_cast<double>(n + 1);
            double remaining_s = per_step * static_cast<double>(nSteps - (n + 1));
            double progress    = static_cast<double>(n + 1) /
                                 static_cast<double>(nSteps);

            const int barWidth = 40;
            int filled = static_cast<int>(progress * barWidth + 0.5);

            std::cout << "\r[";
            for (int i = 0; i < barWidth; ++i){
                std::cout << (i < filled ? '=' : ' ');
            }
            std::cout << "] "
                      << std::setw(6) << std::fixed << std::setprecision(1)
                      << (progress * 100.0) << "% "
                      << "elapsed="  << std::setw(6) << std::setprecision(2) << elapsed_s  << "s "
                      << "eta="      << std::setw(6) << std::setprecision(2) << remaining_s << "s"
                      << std::flush;
        }
    }
    csv_file.close();
    std::cout << "\n";

    dump("Final");
    src::io::write_particles_2d("orbit_particles_final.bin",
                                 species.particles, env.time());
    env.print_timing_summary();

    std::cout << "2D two-particle orbit test done\n";
    std::cout << "Data exported to orbit_data.csv\n";
    return 0;
}
