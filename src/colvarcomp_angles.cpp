// -*- c++ -*-

// This file is part of the Collective Variables module (Colvars).
// The original version of Colvars and its updates are located at:
// https://github.com/Colvars/colvars
// Please update all Colvars source files before making any changes.
// If you wish to distribute your changes, please submit them to the
// Colvars repository at GitHub.

#include "colvarmodule.h"
#include "colvar.h"
#include "colvarcomp.h"


colvar::angle::angle()
{
  set_function_type("angle");
  init_as_angle();

  provide(f_cvc_inv_gradient);
  provide(f_cvc_Jacobian);
  enable(f_cvc_com_based);
}


int colvar::angle::init(std::string const &conf)
{
  int error_code = cvc::init(conf);

  group1 = parse_group(conf, "group1");
  group2 = parse_group(conf, "group2");
  group3 = parse_group(conf, "group3");

  error_code |= init_total_force_params(conf);

  return error_code;
}


colvar::angle::angle(cvm::atom const &a1, cvm::atom const &a2, cvm::atom const &a3) : angle()
{
  group1 = new cvm::atom_group(std::vector<cvm::atom>(1, a1));
  group2 = new cvm::atom_group(std::vector<cvm::atom>(1, a2));
  group3 = new cvm::atom_group(std::vector<cvm::atom>(1, a3));
  register_atom_group(group1);
  register_atom_group(group2);
  register_atom_group(group3);
}


void colvar::angle::calc_value()
{
  cvm::atom_pos const g1_pos = group1->center_of_mass();
  cvm::atom_pos const g2_pos = group2->center_of_mass();
  cvm::atom_pos const g3_pos = group3->center_of_mass();

  r21  = is_enabled(f_cvc_pbc_minimum_image) ?
    cvm::position_distance(g2_pos, g1_pos) :
    g1_pos - g2_pos;
  r21l = r21.norm();
  r23  = is_enabled(f_cvc_pbc_minimum_image) ?
    cvm::position_distance(g2_pos, g3_pos) :
    g3_pos - g2_pos;
  r23l = r23.norm();

  cvm::real const cos_theta = (r21*r23)/(r21l*r23l);

  x.real_value = (180.0/PI) * cvm::acos(cos_theta);
}


void colvar::angle::calc_gradients()
{
  cvm::real const cos_theta = (r21*r23)/(r21l*r23l);
  cvm::real const dxdcos = -1.0 / cvm::sqrt(1.0 - cos_theta*cos_theta);

  dxdr1 = (180.0/PI) * dxdcos *
    (1.0/r21l) * ( r23/r23l + (-1.0) * cos_theta * r21/r21l );

  dxdr3 = (180.0/PI) * dxdcos *
    (1.0/r23l) * ( r21/r21l + (-1.0) * cos_theta * r23/r23l );

  group1->set_weighted_gradient(dxdr1);
  group2->set_weighted_gradient((dxdr1 + dxdr3) * (-1.0));
  group3->set_weighted_gradient(dxdr3);
}


void colvar::angle::calc_force_invgrads()
{
  // This uses a force measurement on groups 1 and 3 only
  // to keep in line with the implicit variable change used to
  // evaluate the Jacobian term (essentially polar coordinates
  // centered on group2, which means group2 is kept fixed
  // when propagating changes in the angle)

  if (is_enabled(f_cvc_one_site_total_force)) {
    group1->read_total_forces();
    cvm::real norm_fact = 1.0 / dxdr1.norm2();
    ft.real_value = norm_fact * dxdr1 * group1->total_force();
  } else {
    group1->read_total_forces();
    group3->read_total_forces();
    cvm::real norm_fact = 1.0 / (dxdr1.norm2() + dxdr3.norm2());
    ft.real_value = norm_fact * ( dxdr1 * group1->total_force()
                                + dxdr3 * group3->total_force());
  }
  return;
}


void colvar::angle::calc_Jacobian_derivative()
{
  // det(J) = (2 pi) r^2 * sin(theta)
  // hence Jd = cot(theta)
  const cvm::real theta = x.real_value * PI / 180.0;
  jd = PI / 180.0 * (theta != 0.0 ? cvm::cos(theta) / cvm::sin(theta) : 0.0);
}



colvar::dipole_angle::dipole_angle()
{
  set_function_type("dipoleAngle");
  init_as_angle();
}


int colvar::dipole_angle::init(std::string const &conf)
{
  int error_code = cvc::init(conf);

  group1 = parse_group(conf, "group1");
  group2 = parse_group(conf, "group2");
  group3 = parse_group(conf, "group3");

  error_code |= init_total_force_params(conf);
  return error_code;
}


void colvar::dipole_angle::calc_value()
{
  cvm::atom_pos const g1_pos = group1->center_of_mass();
  cvm::atom_pos const g2_pos = group2->center_of_mass();
  cvm::atom_pos const g3_pos = group3->center_of_mass();

  group1->calc_dipole(g1_pos);

  r21 = group1->dipole();
  r21l = r21.norm();
  r23  = is_enabled(f_cvc_pbc_minimum_image) ?
    cvm::position_distance(g2_pos, g3_pos) :
    g3_pos - g2_pos;
  r23l = r23.norm();

  cvm::real const cos_theta = (r21*r23)/(r21l*r23l);

  x.real_value = (180.0/PI) * cvm::acos(cos_theta);
}

//to be implemented
//void colvar::dipole_angle::calc_force_invgrads(){}
//void colvar::dipole_angle::calc_Jacobian_derivative(){}

void colvar::dipole_angle::calc_gradients()
{
  cvm::real const cos_theta = (r21*r23)/(r21l*r23l);
  cvm::real const dxdcos = -1.0 / cvm::sqrt(1.0 - cos_theta*cos_theta);

  dxdr1 = (180.0/PI) * dxdcos *
  (1.0/r21l)* (r23/r23l + (-1.0) * cos_theta * r21/r21l );

  dxdr3 =  (180.0/PI) * dxdcos *
    (1.0/r23l) * ( r21/r21l + (-1.0) * cos_theta * r23/r23l );

  //this auxiliar variables are to avoid numerical errors inside "for"
  double aux1 = group1->total_charge/group1->total_mass;
  // double aux2 = group2->total_charge/group2->total_mass;
  // double aux3 = group3->total_charge/group3->total_mass;

  size_t i;
  for (i = 0; i < group1->size(); i++) {
    (*group1)[i].grad =((*group1)[i].charge + (-1)* (*group1)[i].mass * aux1) * (dxdr1);
  }

  for (i = 0; i < group2->size(); i++) {
    (*group2)[i].grad = ((*group2)[i].mass/group2->total_mass)* dxdr3 * (-1.0);
  }

  for (i = 0; i < group3->size(); i++) {
    (*group3)[i].grad =((*group3)[i].mass/group3->total_mass) * (dxdr3);
  }
}



colvar::dihedral::dihedral()
{
  set_function_type("dihedral");
  init_as_periodic_angle();
  provide(f_cvc_inv_gradient);
  provide(f_cvc_Jacobian);
  enable(f_cvc_com_based);
}


int colvar::dihedral::init(std::string const &conf)
{
  int error_code = cvc::init(conf);

  group1 = parse_group(conf, "group1");
  group2 = parse_group(conf, "group2");
  group3 = parse_group(conf, "group3");
  group4 = parse_group(conf, "group4");

  error_code |= init_total_force_params(conf);
  return error_code;
}


colvar::dihedral::dihedral(cvm::atom const &a1, cvm::atom const &a2, cvm::atom const &a3,
                           cvm::atom const &a4)
  : dihedral()
{
  b_1site_force = false;

  group1 = new cvm::atom_group(std::vector<cvm::atom>(1, a1));
  group2 = new cvm::atom_group(std::vector<cvm::atom>(1, a2));
  group3 = new cvm::atom_group(std::vector<cvm::atom>(1, a3));
  group4 = new cvm::atom_group(std::vector<cvm::atom>(1, a4));
  register_atom_group(group1);
  register_atom_group(group2);
  register_atom_group(group3);
  register_atom_group(group4);
}


void colvar::dihedral::calc_value()
{
  cvm::atom_pos const g1_pos = group1->center_of_mass();
  cvm::atom_pos const g2_pos = group2->center_of_mass();
  cvm::atom_pos const g3_pos = group3->center_of_mass();
  cvm::atom_pos const g4_pos = group4->center_of_mass();

  // Usual sign convention: r12 = r2 - r1
  r12 = is_enabled(f_cvc_pbc_minimum_image) ?
    cvm::position_distance(g1_pos, g2_pos) :
    g2_pos - g1_pos;
  r23 = is_enabled(f_cvc_pbc_minimum_image) ?
    cvm::position_distance(g2_pos, g3_pos) :
    g3_pos - g2_pos;
  r34 = is_enabled(f_cvc_pbc_minimum_image) ?
    cvm::position_distance(g3_pos, g4_pos) :
    g4_pos - g3_pos;

  cvm::rvector const n1 = cvm::rvector::outer(r12, r23);
  cvm::rvector const n2 = cvm::rvector::outer(r23, r34);

  cvm::real const cos_phi = n1 * n2;
  cvm::real const sin_phi = n1 * r34 * r23.norm();

  x.real_value = (180.0/PI) * cvm::atan2(sin_phi, cos_phi);
  wrap(x);
}


void colvar::dihedral::calc_gradients()
{
  cvm::rvector A = cvm::rvector::outer(r12, r23);
  cvm::real   rA = A.norm();
  cvm::rvector B = cvm::rvector::outer(r23, r34);
  cvm::real   rB = B.norm();
  cvm::rvector C = cvm::rvector::outer(r23, A);
  cvm::real   rC = C.norm();

  cvm::real const cos_phi = (A*B)/(rA*rB);
  cvm::real const sin_phi = (C*B)/(rC*rB);

  cvm::rvector f1, f2, f3;

  rB = 1.0/rB;
  B *= rB;

  if (cvm::fabs(sin_phi) > 0.1) {
    rA = 1.0/rA;
    A *= rA;
    cvm::rvector const dcosdA = rA*(cos_phi*A-B);
    cvm::rvector const dcosdB = rB*(cos_phi*B-A);
    // rA = 1.0;

    cvm::real const K = (1.0/sin_phi) * (180.0/PI);

        f1 = K * cvm::rvector::outer(r23, dcosdA);
        f3 = K * cvm::rvector::outer(dcosdB, r23);
        f2 = K * (cvm::rvector::outer(dcosdA, r12)
                   +  cvm::rvector::outer(r34, dcosdB));
  }
  else {
    rC = 1.0/rC;
    C *= rC;
    cvm::rvector const dsindC = rC*(sin_phi*C-B);
    cvm::rvector const dsindB = rB*(sin_phi*B-C);
    // rC = 1.0;

    cvm::real    const K = (-1.0/cos_phi) * (180.0/PI);

    f1.x = K*((r23.y*r23.y + r23.z*r23.z)*dsindC.x
              - r23.x*r23.y*dsindC.y
              - r23.x*r23.z*dsindC.z);
    f1.y = K*((r23.z*r23.z + r23.x*r23.x)*dsindC.y
              - r23.y*r23.z*dsindC.z
              - r23.y*r23.x*dsindC.x);
    f1.z = K*((r23.x*r23.x + r23.y*r23.y)*dsindC.z
              - r23.z*r23.x*dsindC.x
              - r23.z*r23.y*dsindC.y);

    f3 = cvm::rvector::outer(dsindB, r23);
    f3 *= K;

    f2.x = K*(-(r23.y*r12.y + r23.z*r12.z)*dsindC.x
              +(2.0*r23.x*r12.y - r12.x*r23.y)*dsindC.y
              +(2.0*r23.x*r12.z - r12.x*r23.z)*dsindC.z
              +dsindB.z*r34.y - dsindB.y*r34.z);
    f2.y = K*(-(r23.z*r12.z + r23.x*r12.x)*dsindC.y
              +(2.0*r23.y*r12.z - r12.y*r23.z)*dsindC.z
              +(2.0*r23.y*r12.x - r12.y*r23.x)*dsindC.x
              +dsindB.x*r34.z - dsindB.z*r34.x);
    f2.z = K*(-(r23.x*r12.x + r23.y*r12.y)*dsindC.z
              +(2.0*r23.z*r12.x - r12.z*r23.x)*dsindC.x
              +(2.0*r23.z*r12.y - r12.z*r23.y)*dsindC.y
              +dsindB.y*r34.x - dsindB.x*r34.y);
  }

  group1->set_weighted_gradient(-f1);
  group2->set_weighted_gradient(-f2 + f1);
  group3->set_weighted_gradient(-f3 + f2);
  group4->set_weighted_gradient(f3);
}


void colvar::dihedral::calc_force_invgrads()
{
  cvm::rvector const u12 = r12.unit();
  cvm::rvector const u23 = r23.unit();
  cvm::rvector const u34 = r34.unit();

  cvm::real const d12 = r12.norm();
  cvm::real const d34 = r34.norm();

  cvm::rvector const cross1 = (cvm::rvector::outer(u23, u12)).unit();
  cvm::rvector const cross4 = (cvm::rvector::outer(u23, u34)).unit();

  cvm::real const dot1 = u23 * u12;
  cvm::real const dot4 = u23 * u34;

  cvm::real const fact1 = d12 * cvm::sqrt(1.0 - dot1 * dot1);
  cvm::real const fact4 = d34 * cvm::sqrt(1.0 - dot4 * dot4);

  group1->read_total_forces();
  if (is_enabled(f_cvc_one_site_total_force)) {
    // This is only measuring the force on group 1
    ft.real_value = PI/180.0 * fact1 * (cross1 * group1->total_force());
  } else {
    // Default case: use groups 1 and 4
    group4->read_total_forces();
    ft.real_value = PI/180.0 * 0.5 * (fact1 * (cross1 * group1->total_force())
                                      + fact4 * (cross4 * group4->total_force()));
  }
}


void colvar::dihedral::calc_Jacobian_derivative()
{
  // With this choice of inverse gradient ("internal coordinates"), Jacobian correction is 0
  jd = 0.0;
}



colvar::polar_theta::polar_theta()
{
  r = theta = phi = 0.0;
  set_function_type("polarTheta");
  enable(f_cvc_com_based);
  init_as_angle();
}


int colvar::polar_theta::init(std::string const &conf)
{
  int error_code = cvc::init(conf);
  atoms = parse_group(conf, "atoms");
  error_code |= init_total_force_params(conf);
  return error_code;
}


void colvar::polar_theta::calc_value()
{
  cvm::rvector pos = atoms->center_of_mass();
  r = atoms->center_of_mass().norm();
  // Internal values of theta and phi are radians
  theta = (r > 0.) ? cvm::acos(pos.z / r) : 0.;
  phi = cvm::atan2(pos.y, pos.x);
  x.real_value = (180.0/PI) * theta;
}


void colvar::polar_theta::calc_gradients()
{
  if (r == 0.)
    atoms->set_weighted_gradient(cvm::rvector(0., 0., 0.));
  else
    atoms->set_weighted_gradient(cvm::rvector(
      (180.0/PI) *  cvm::cos(theta) * cvm::cos(phi) / r,
      (180.0/PI) *  cvm::cos(theta) * cvm::sin(phi) / r,
      (180.0/PI) * -cvm::sin(theta) / r));
}



colvar::polar_phi::polar_phi()
{
  r = theta = phi = 0.0;
  set_function_type("polarPhi");
  enable(f_cvc_com_based);
  init_as_periodic_angle();
}


int colvar::polar_phi::init(std::string const &conf)
{
  int error_code = cvc::init(conf);
  atoms = parse_group(conf, "atoms");
  error_code |= init_total_force_params(conf);
  return error_code;
}


void colvar::polar_phi::calc_value()
{
  cvm::rvector pos = atoms->center_of_mass();
  r = atoms->center_of_mass().norm();
  // Internal values of theta and phi are radians
  theta = (r > 0.) ? cvm::acos(pos.z / r) : 0.;
  phi = cvm::atan2(pos.y, pos.x);
  x.real_value = (180.0/PI) * phi;
}


void colvar::polar_phi::calc_gradients()
{
  atoms->set_weighted_gradient(cvm::rvector(
    (180.0/PI) * -cvm::sin(phi) / (r*cvm::sin(theta)),
    (180.0/PI) *  cvm::cos(phi) / (r*cvm::sin(theta)),
    0.));
}
