int teukolsky_wave_gauge;
int multipole_l;
int multipole_m;
int multipole_parity;
int id_dynamics_type;
int id_functional_form;
int kk;
double sin_theta;
double cos_theta;
double sin_2theta;
double cos_2theta;
double sin_3theta;
double cos_3theta;
double sin_4theta;
double cos_4theta;
double sin_phi;
double cos_phi;
double sin_2phi;
double cos_2phi;
double sin_3phi;
double cos_3phi;
double sin_4phi;
double cos_4phi;
double rho_sqrd;
double rho;
double rho_axis;
double rr;
double r_origin;
double r_sqrd;
double a_0;
double b_0;
double r_0;
double r_0_sqrd;
double width_teuk;
double aa;
double bb;
double amp_teuk;
double f0;
double df0;
double ddf0;
double d3f0;
double d4f0;
double d5f0;
double d6f0;
double f1;
double f2;
double df1;
double df2;
double ddf1;
double ddf2;
double d3f1;
double d3f2;
double d4f1;
double d4f2;
double d5f1;
double d5f2;
double d6f1;
double d6f2;
double  A_lm;
double  B_lm;
double  C_lm;
double  K_lm;
double  L_lm;
double drdx;
double drdy;
double drdz;
double dthdx;
double dthdy;
double dthdz;
double dphdx;
double dphdy;
double dphdz;
double g_xx;
double g_xy;
double g_xz;
double g_yy;
double g_yz;
double g_zz;
double epsilon_sq;
double Ylm;
double Ylm_theta;
double Ylm_phi;
double Ylm_thth;
double Ylm_thphi;
double Y20;
double Y21;
double Y2n1;
double Y22;
double Y2n2;
double Y20_theta;
double Y21_theta;
double Y2n1_theta;
double Y22_theta;
double Y2n2_theta;
double Y20_phi;
double Y21_phi;
double Y2n1_phi;
double Y22_phi;
double Y2n2_phi;
double Y20_thth;
double Y21_thth;
double Y2n1_thth;
double Y22_thth;
double Y2n2_thth;
double Y20_thphi;
double Y21_thphi;
double Y2n1_thphi;
double Y22_thphi;
double Y2n2_thphi;
double Y30;
double Y31;
double Y3n1;
double Y32;
double Y3n2;
double Y33;
double Y3n3;
double Y30_theta;
double Y31_theta;
double Y3n1_theta;
double Y32_theta;
double Y3n2_theta;
double Y33_theta;
double Y3n3_theta;
double Y30_phi;
double Y31_phi;
double Y3n1_phi;
double Y32_phi;
double Y3n2_phi;
double Y33_phi;
double Y3n3_phi;
double Y30_thth;
double Y31_thth;
double Y3n1_thth;
double Y32_thth;
double Y3n2_thth;
double Y33_thth;
double Y3n3_thth;
double Y30_thphi;
double Y31_thphi;
double Y3n1_thphi;
double Y32_thphi;
double Y3n2_thphi;
double Y33_thphi;
double Y3n3_thphi;
double Y40;
double Y41;
double Y4n1;
double Y42;
double Y4n2;
double Y43;
double Y4n3;
double Y44;
double Y4n4;
double Y40_theta;
double Y41_theta;
double Y4n1_theta;
double Y42_theta;
double Y4n2_theta;
double Y43_theta;
double Y4n3_theta;
double Y44_theta;
double Y4n4_theta;
double Y40_phi;
double Y41_phi;
double Y4n1_phi;
double Y42_phi;
double Y4n2_phi;
double Y43_phi;
double Y4n3_phi;
double Y44_phi;
double Y4n4_phi;
double Y40_thth;
double Y41_thth;
double Y4n1_thth;
double Y42_thth;
double Y4n2_thth;
double Y43_thth;
double Y4n3_thth;
double Y44_thth;
double Y4n4_thth;
double Y40_thphi;
double Y41_thphi;
double Y4n1_thphi;
double Y42_thphi;
double Y4n2_thphi;
double Y43_thphi;
double Y4n3_thphi;
double Y44_thphi;
double Y4n4_thphi;
double h_bare_rr;
double h_bare_rth;
double h_bare_rph;
double h_bare_thth;
double h_bare_thph;
double h_bare_phph;
double g_bare_rr;
double g_bare_rth;
double g_bare_rph;
double g_bare_thth;
double g_bare_thph;
double g_bare_phph;
double g_rr;
double g_rth;
double g_rph;
double g_thth;
double g_thph;
double g_phph;
double determinant;
double pi;
double fr0;
//-----------------------------------------------------------
//
// Teukolsky (quadrupole) waves
//
// This is a first attempt at coding up linearized quadrupole
// waves.  As a first go, we will make some simplifying
// assumptions.  In particular, we assume time symmetric
// initial data.  We will assume an l=2, m=0 multipole.  We
// will also work in transverse-traceless (TT) gauge.  (This
// is the gauge that Teukolsky and Rinne work in.)  We also
// work only with the even parity wave.
//
//-----------------------------------------------------------

// We may eventually want a variety of options for the kinds
// of waves that we want so we may as well define them here.

// Choose the gauge in which we will work.  Choices include
// TT gauge (0, default) and the "LATE" gauge of
// Lorentz-Thorne-Abrahams-Evans (1)
teukolsky_wave_gauge = bssn::TEUK_GAUGE;

// Choose the multipole order:  (l,m).  Choices include
// quadrupole (l=2, default), octopole (l=3), and hexadecapole
// (l=4) with all the possible m values (m=0, default).
multipole_l = bssn::TEUK_L_MODE;
multipole_m = bssn::TEUK_M_MODE;

// Choose the polarization state.  Choices are either even
// (0, default) or odd (1) parity waves.
multipole_parity = bssn::MULTIPOLE_PARITY;

// Choose the dynamics or time behavior of the initial data.
// Choices include time symmetric (0, default), ingoing (1),
// outgoing (2), time anti-symmetric (3) and general (4).
id_dynamics_type = bssn::TEUK_ID_DYNAMICS_TYPE;

// Choose the functional form of the initial pulse.  Choices
// include a C^k polynomial with compact support (0, default)
// a Gaussian (1), a bump function (2), and Abrahams and Evans"
// Hermite polynomial modulated Gaussian (3).
id_functional_form = bssn::TEUK_ID_FUNCTIONAL_FORM;

// The amplitude, (half-)width and center of the initial data:
amp_teuk = bssn::TEUK_AMP;
width_teuk = bssn::TEUK_WIDTH; //half width really; width is twice this
r_0 = bssn::TEUK_R_0;
// The order of differentiability of the C^k polynomials with
// compact support.
kk = bssn::TEUK_KK;

if ( id_dynamics_type == 0 ) { // time symmetric
	a_0 = 1.0;
	b_0 = 0.0;
} else if ( id_dynamics_type == 1) { // ingoing
	a_0 = 1.0;
	b_0 = 1.0;
} else if ( id_dynamics_type == 2 ) { // outgoing
	a_0 = 1.0;
	b_0 = -1.0;
} else if ( id_dynamics_type == 3 ) { // time anti-symmetric
	a_0 = 0.0;
	b_0 = 1.0;
} else {
	std::cout << "this initial data choice is not yet allowed/n";
}

pi = 3.1415926535897932384626433832795028841971693993751058209749445923078164062862089986;

//std::cout << "starting the loop in initial data ... /n";

rho_sqrd = xx*xx + yy*yy;
rho = sqrt( rho_sqrd );

r_sqrd = rho_sqrd + zz*zz;
rr = sqrt( r_sqrd );

epsilon_sq = 1.0e-8;    // maybe we can read in grid info later? 
r_origin = sqrt( r_sqrd + epsilon_sq );

// Away from the axis of symmetry
if ( abs(xx) > 1.0e-7  ||  abs(yy) > 1.0e-7 ) {
	sin_theta = rho / rr;
	cos_theta = zz / rr;
	sin_2theta = 2.0 * sin_theta * cos_theta;
	cos_2theta = pow(cos_theta,2)-pow(sin_theta,2);
	sin_3theta = ( 3.0 * sin_theta ) - ( 4.0 * pow(sin_theta,3) );
	cos_3theta = ( 4.0 * pow(cos_theta,3) )-( 3.0 * cos_theta );
	sin_4theta = cos_theta * (4.0 * sin_theta - 8.0 * pow(sin_theta,3));
	cos_4theta = 8.0 * pow(cos_theta,4) - 8.0 * pow(cos_theta,2) + 1.0;
	sin_phi = yy / rho;
	cos_phi = xx / rho;
	sin_2phi = 2.0 * sin_phi * cos_phi;
	cos_2phi = pow(cos_phi,2) - pow(sin_phi,2);
	sin_3phi = ( 3.0 * sin_phi ) - ( 4.0 * pow(sin_phi,3) );
	cos_3phi = ( 4.0 * pow(cos_phi,3) ) - ( 3.0 * cos_phi );
	sin_4phi = cos_phi * (4.0 * sin_phi - 8.0 * pow(sin_phi,3));
	cos_4phi = 8.0 * pow(cos_phi,4) - 8.0 * pow(cos_phi,2) + 1.0;
} 
// On the axis of symmetry but away from the origin
else if ( ( abs(xx) < 1.0e-7 && abs(yy) < 1.0e-7 ) && abs(zz) > 1.0e-7  ) {
	sin_theta = rho / rr;
	cos_theta = zz / rr;
	sin_2theta = 2.0 * sin_theta * cos_theta;
	cos_2theta = pow(cos_theta,2)-pow(sin_theta,2);
	sin_3theta = ( 3.0 * sin_theta ) - ( 4.0 * pow(sin_theta,3) );
	cos_3theta = ( 4.0 * pow(cos_theta,3) )-( 3.0 * cos_theta );
	sin_4theta = cos_theta * (4.0 * sin_theta - 8.0 * pow(sin_theta,3));
	cos_4theta = 8.0 * pow(cos_theta,4) - 8.0 * pow(cos_theta,2) + 1.0;
	sin_phi = 0.0;
	cos_phi = 0.0;
	sin_2phi = 2.0 * sin_phi * cos_phi;
	cos_2phi = pow(cos_phi,2) - pow(sin_phi,2);
	sin_3phi = ( 3.0 * sin_phi ) - ( 4.0 * pow(sin_phi,3) );
	cos_3phi = ( 4.0 * pow(cos_phi,3) ) - ( 3.0 * cos_phi );
	sin_4phi = cos_phi * (4.0 * sin_phi - 8.0 * pow(sin_phi,3));
	cos_4phi = 8.0 * pow(cos_phi,4) - 8.0 * pow(cos_phi,2) + 1.0;
}
// At the origin
else if ( ( abs(xx) < 1.0e-7 && abs(yy) < 1.0e-7 ) && abs(zz) < 1.0e-7 ) {
	sin_theta = 0.0;
	cos_theta = 0.0;
	sin_2theta = 2.0 * sin_theta * cos_theta;
	cos_2theta = pow(cos_theta,2)-pow(sin_theta,2);
	sin_3theta = ( 3.0 * sin_theta ) - ( 4.0 * pow(sin_theta,3) );
	cos_3theta = ( 4.0 * pow(cos_theta,3) )-( 3.0 * cos_theta );
	sin_4theta = cos_theta * (4.0 * sin_theta - 8.0 * pow(sin_theta,3));
	cos_4theta = 8.0 * pow(cos_theta,4) - 8.0 * pow(cos_theta,2) + 1.0;
	sin_phi = 0.0;
	cos_phi = 0.0;
	sin_2phi = 2.0*sin_phi*cos_phi;
	cos_2phi = pow(cos_phi,2) - pow(sin_phi,2);
	sin_3phi = ( 3.0 * sin_phi ) - ( 4.0 * pow(sin_phi,3) );
	cos_3phi = ( 4.0 * pow(cos_phi,3) ) - ( 3.0 * cos_phi );
	sin_4phi = cos_phi * (4.0 * sin_phi - 8.0 * pow(sin_phi,3));
	cos_4phi = 8.0 * pow(cos_phi,4) - 8.0 * pow(cos_phi,2) + 1.0;
} else {
	std::cout <<" problem in initial data /n";
	std::cout << " x = " << xx ;
	std::cout <<" y = "<< yy ;
	std::cout <<" z = "<< zz;
	std::cout << " r = "<< rr;
	std::cout << " rho = "<< rho;
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////            
//building the initial radial function
//
if (id_functional_form == 0) { // C^k polynomail with compact support
	aa = r_0 - width_teuk; // leftmost zero of initial data
	bb = r_0 + width_teuk; // rightmost zero of initial data

	if ( aa < 0.0 || bb < 0.0 ) {
	std::cout<<"initial data extends to r<0";
	}	

	if ( rr >= aa && rr <= bb ) {
		fr0 = pow((bb - r_0),(kk+1))*pow((bb - r_0),(kk+1));
		
		f1 = pow((rr - aa),(kk+1));
		f2 = pow((bb - rr),(kk+1));

		df1 = (kk+1)*pow((rr-aa),kk);
		df2 = - (kk+1)*pow((bb-rr),kk);
	
		ddf1 = (kk+1)*kk*pow((rr-aa),(kk-1));
		ddf2 = (kk+1)*kk*pow((bb-rr),(kk-1));
	
		d3f1 = (kk+1)*kk*(kk-1)*pow((rr-aa),(kk-2));
		d3f2 = - (kk+1)*kk*(kk-1)*pow((bb-rr),(kk-2));
	
		d4f1 = (kk+1)*kk*(kk-1)*(kk-2)*pow((rr-aa),(kk-3));
		d4f2 = (kk+1)*kk*(kk-1)*(kk-2)*pow((bb-rr),(kk-3));
	
		d5f1 = (kk+1)*kk*(kk-1)*(kk-2)*(kk-3)*pow((rr-aa),(kk-4));
		d5f2 = (kk+1)*kk*(kk-1)*(kk-2)*(kk-3)*pow((bb-rr),(kk-4));

		d6f1 = (kk+1)*kk*(kk-1)*(kk-2)*(kk-3)*(kk-4)*pow((rr-aa),(kk-5));
		d6f2 = (kk+1)*kk*(kk-1)*(kk-2)*(kk-3)*(kk-4)*pow((bb-rr),(kk-5));
	
		f0 = (f1 * f2) / fr0;
	
		df0 = (df1*f2 + f1*df2) / fr0;
	
		ddf0 = (ddf1*f2 + 2.0*df1*df2 + f1*ddf2) / fr0;
	
		d3f0 = (d3f1*f2 + 3.0*ddf1*df2 + 3.0*df1*ddf2 + f1*d3f2) / fr0 ;
	
		d4f0 = (d4f1*f2 + 4.0 * d3f1*df2 + 6.0 * ddf1*ddf2 + 4.0 * df1*d3f2 + f1*d4f2) / fr0;
	
		d5f0 = (d5f1*f2 + 5.0 * d4f1*df2 + 10.0 * d3f1*ddf2 + 10.0 * ddf1*d3f2 +  5.0 * df1*d4f2 + f1*d5f1) / fr0;
	
		d6f0 = (d6f1*f2 + 6.0 * d5f1*df2 + 15.0 * d4f1*ddf2 + 20.0 * d3f1*d3f2 + 15.0 * ddf1*d4f2 +  6.0 * df1*d5f2 + f1*d6f2) / fr0;
	} else {
		f0 = 0.0 ;
		df0 = 0.0 ;
		ddf0 = 0.0;
		d3f0 = 0.0;
		d4f0 = 0.0;
		d5f0 = 0.0;
		d6f0 = 0.0;
	}
} else if (id_functional_form == 1) { //Guassian
	f0 = exp( - pow(rr-r_0,2) / width_teuk );
	df0 = ( -2.0 * (rr-r_0) / width_teuk ) * f0;
	ddf0 = ( (4.0 * pow(rr-r_0, 2) - 2.0 * width_teuk) / pow(width_teuk,2) ) * f0;
	d3f0 = ( (-8.0 * pow(rr-r_0, 3) + 12.0 * (rr-r_0) * width_teuk ) / pow(width_teuk,3) ) * f0;
	d4f0 = ( (16.0 * pow(rr-r_0, 4) -48.0 * pow(rr-r_0,2) * width_teuk + 12.0 * pow(width_teuk,2) ) / pow(width_teuk,4) ) * f0;
	d5f0 = (8.0*(rr-r_0)*(-4.0 * pow(rr-r_0,4) + 20.0 * pow(rr-r_0,2) * width_teuk - 15.0 * pow(width_teuk,2) / pow(width_teuk,5) ) ) * f0;
	d6f0 = (8.0 * (8.0 * pow(rr-r_0,6) - 60.0 * pow(rr-r_0,4) * width_teuk + 90.0 * pow(rr-r_0,2) * pow(width_teuk,2) - 15.0 * pow(width_teuk,3) ) / pow(width_teuk,6) ) * f0;
}
else if (id_functional_form == 2){//Bump function
if ( rr > r_0 - width_teuk/2 + 0.5 && rr < r_0 + width_teuk/2 -0.5)// to avoid having an indeterminant exponent I removed the equal to. 
{
	double bump1 = r_0 - width_teuk/2; 
	double bump2 = r_0 + width_teuk/2;
	f0 = exp(1/((rr- bump1)* (rr -bump2)));

	df0 = f0*(bump1+bump2 -2 *rr)/(pow((rr-bump1),2)*pow((rr-bump2),2));

	ddf0 = ((exp(1 / ((-bump1 + rr) * (-bump2 + rr))) *
                        (pow(bump1, 2) + 2 * bump1 * bump2 +
                         2 * pow(bump1, 3) * bump2 + pow(bump2, 2) +
                         2 * pow(bump1, 2) * pow(bump2, 2) +
                         2 * bump1 * pow(bump2, 3) -
                         2 * (bump1 + bump2) *
                             (2 + pow(bump1, 2) + 4 * bump1 * bump2 + pow(bump2, 2)) * rr +
                         4 * (1 + (2 * bump1 + bump2) * (bump1 + 2 * bump2)) * pow(rr, 2) -
                         12 * (bump1 + bump2) * pow(rr, 3) + 6 * pow(rr, 4))))/(pow((bump1 - rr), 4) * pow((bump2 - rr), 4));

d3f0 = ((exp(1 / ((-bump1 + rr) * (-bump2 + rr))) *
                        ((pow((bump1 + bump2 - 2 * rr), 3)) +
                         6 * pow((bump1 - rr), 5) * pow((bump2 - rr), 2) +
                         6 * pow((bump1 - rr), 4) * pow((bump2 - rr), 3) +
                         6 * pow((bump1 - rr), 3) * pow((bump2 - rr), 4) +
                         6 * pow((bump1 - rr), 2) * pow((bump2 - rr), 5) +
                         6 * (bump1 + bump2 - 2 * rr) * (bump1 - rr) * (bump2 - rr) *
                             (pow(bump1, 2) + bump1 * bump2 + pow(bump2, 2) -
                              3 * (bump1 + bump2) * rr + 3 * pow(rr, 2)))))/(pow((bump1 - rr), 6) * pow((bump2 - rr), 6));

	d4f0 = ((1 / (pow((bump1 - rr), 8) * pow((bump2 - rr), 8))) *
                       exp(1 / ((-bump1 + rr) * (-bump2 + rr))) *
                       (pow((bump1 - rr), 4) + 6 * pow((bump1 - rr), 2) * pow((bump2 - rr), 2) +
                        36 * pow((bump1 - rr), 4) * pow((bump2 - rr), 2) +
                        36 * pow((bump1 - rr), 6) * pow((bump2 - rr), 2) +
                        pow((bump2 - rr), 4) + 36 * pow((bump1 - rr), 2) * pow((bump2 - rr), 4) +
                        84 * pow((bump1 - rr), 4) * pow((bump2 - rr), 4) +
                        24 * pow((bump1 - rr), 6) * pow((bump2 - rr), 4) +
                        36 * pow((bump1 - rr), 2) * pow((bump2 - rr), 6) +
                        24 * pow((bump1 - rr), 4) * pow((bump2 - rr), 6) +
                        4 * pow((-bump1 + rr), 3) * (-bump2 + rr) +
                        12 * pow((-bump1 + rr), 5) * (-bump2 + rr) +
                        4 * (-bump1 + rr) * pow((-bump2 + rr), 3) +
                        48 * pow((-bump1 + rr), 3) * pow((-bump2 + rr), 3) +
                        72 * pow((-bump1 + rr), 5) * pow((-bump2 + rr), 3) +
                        24 * pow((-bump1 + rr), 7) * pow((-bump2 + rr), 3) +
                        12 * (-bump1 + rr) * pow((-bump2 + rr), 5) +
                        72 * pow((-bump1 + rr), 3) * pow((-bump2 + rr), 5) +
                        24 * pow((-bump1 + rr), 5) * pow((-bump2 + rr), 5) +
                        24 * pow((-bump1 + rr), 3) * pow((-bump2 + rr), 7)))/(pow((bump1 - rr), 8) * pow((bump2 - rr), 8));
	d5f0 = ((1 / (pow((bump1 - rr), 10) * pow((bump2 - rr), 10))) *
                       exp(1 / ((-bump1 + rr) * (-bump2 + rr))) *
                       (pow((bump1 - rr), 5) + 5 * pow((bump1 - rr), 4) * (bump2 - rr) +
                        20 * pow((bump1 - rr), 6) * (bump2 - rr) +
                        10 * pow((bump1 - rr), 3) * pow((bump2 - rr), 2) +
                        80 * pow((bump1 - rr), 5) * pow((bump2 - rr), 2) +
                        120 * pow((bump1 - rr), 7) * pow((bump2 - rr), 2) +
                        10 * pow((bump1 - rr), 2) * pow((bump2 - rr), 3) +
                        5 * (bump1 - rr) * pow((bump2 - rr), 4) +
                        140 * pow((bump1 - rr), 3) * pow((bump2 - rr), 4) +
                        540 * pow((bump1 - rr), 5) * pow((bump2 - rr), 4) +
                        480 * pow((bump1 - rr), 7) * pow((bump2 - rr), 4) +
                        120 * pow((bump1 - rr), 9) * pow((bump2 - rr), 4) +
                        pow((bump2 - rr), 5) +
                        80 * pow((bump1 - rr), 2) * pow((bump2 - rr), 5) +
                        20 * (bump1 - rr) * pow((bump2 - rr), 6) +
                        360 * pow((bump1 - rr), 3) * pow((bump2 - rr), 6) +
                        600 * pow((bump1 - rr), 5) * pow((bump2 - rr), 6) +
                        120 * pow((bump1 - rr), 7) * pow((bump2 - rr), 6) +
                        240 * pow((bump1 - rr), 3) * pow((bump2 - rr), 8) +
                        120 * pow((bump1 - rr), 5) * pow((bump2 - rr), 8) -
                        140 * pow((bump1 - rr), 4) * (-bump2 + rr) * pow((-bump2 + rr), 3) -
                        360 * pow((bump1 - rr), 6) * (-bump2 + rr) * pow((-bump2 + rr), 3) -
                        240 * pow((bump1 - rr), 8) * (-bump2 + rr) * pow((-bump2 + rr), 3) -
                        540 * pow((bump1 - rr), 4) * (-bump2 + rr) * pow((-bump2 + rr), 5) -
                        600 * pow((bump1 - rr), 6) * (-bump2 + rr) * pow((-bump2 + rr), 5) -
                        120 * pow((bump1 - rr), 8) * (-bump2 + rr) * pow((-bump2 + rr), 5) -
                        120 * pow((bump1 - rr), 2) * (-bump2 + rr) * pow((-bump2 + rr), 7) -
                        480 * pow((bump1 - rr), 4) * (-bump2 + rr) * pow((-bump2 + rr), 7) -
                        120 * pow((bump1 - rr), 6) * (-bump2 + rr) * pow((-bump2 + rr), 7) -
                        120 * pow((bump1 - rr), 4) * (-bump2 + rr) * pow((-bump2 + rr), 9)))/(pow((bump1 - rr), 10) * pow((bump2 - rr), 10));

	d6f0 = ((1 / (pow((bump1 - rr), 12) * pow((bump2 - rr), 12))) *
                       exp(1 / ((-bump1 + rr) * (-bump2 + rr))) *
                       (pow((bump1 - rr), 6) + 15 * pow((bump1 - rr), 4) * pow((bump2 - rr), 2) +
                        150 * pow((bump1 - rr), 6) * pow((bump2 - rr), 2) +
                        300 * pow((bump1 - rr), 8) * pow((bump2 - rr), 2) +
                        15 * pow((bump1 - rr), 2) * pow((bump2 - rr), 4) +
                        420 * pow((bump1 - rr), 4) * pow((bump2 - rr), 4) +
                        2280 * pow((bump1 - rr), 6) * pow((bump2 - rr), 4) +
                        3600 * pow((bump1 - rr), 8) * pow((bump2 - rr), 4) +
                        1800 * pow((bump1 - rr), 10) * pow((bump2 - rr), 4) +
                        pow((bump2 - rr), 6) +
                        150 * pow((bump1 - rr), 2) * pow((bump2 - rr), 6) +
                        2280 * pow((bump1 - rr), 4) * pow((bump2 - rr), 6) +
                        6600 * pow((bump1 - rr), 6) * pow((bump2 - rr), 6) +
                        4680 * pow((bump1 - rr), 8) * pow((bump2 - rr), 6) +
                        720 * pow((bump1 - rr), 10) * pow((bump2 - rr), 6) +
                        300 * pow((bump1 - rr), 2) * pow((bump2 - rr), 8) +
                        3600 * pow((bump1 - rr), 4) * pow((bump2 - rr), 8) +
                        4680 * pow((bump1 - rr), 6) * pow((bump2 - rr), 8) +
                        720 * pow((bump1 - rr), 8) * pow((bump2 - rr), 8) +
                        1800 * pow((bump1 - rr), 4) * pow((bump2 - rr), 10) +
                        720 * pow((bump1 - rr), 6) * pow((bump2 - rr), 10) +
                        6 * pow((-bump1 + rr), 5) * (-bump2 + rr) +
                        30 * pow((-bump1 + rr), 7) * (-bump2 + rr) +
                        20 * pow((-bump1 + rr), 3) * pow((-bump2 + rr), 3) +
                        330 * pow((-bump1 + rr), 5) * pow((-bump2 + rr), 3) +
                        1200 * pow((-bump1 + rr), 7) * pow((-bump2 + rr), 3) +
                        1200 * pow((-bump1 + rr), 9) * pow((-bump2 + rr), 3) +
                        6 * (-bump1 + rr) * pow((-bump2 + rr), 5) +
                        330 * pow((-bump1 + rr), 3) * pow((-bump2 + rr), 5) +
                        2760 * pow((-bump1 + rr), 5) * pow((-bump2 + rr), 5) +
                        5760 * pow((-bump1 + rr), 7) * pow((-bump2 + rr), 5) +
                        3600 * pow((-bump1 + rr), 9) * pow((-bump2 + rr), 5) +
                        720 * pow((-bump1 + rr), 11) * pow((-bump2 + rr), 5) +
                        30 * (-bump1 + rr) * pow((-bump2 + rr), 7) +
                        1200 * pow((-bump1 + rr), 3) * pow((-bump2 + rr), 7) +
                        5760 * pow((-bump1 + rr), 5) * pow((-bump2 + rr), 7) +
                        5040 * pow((-bump1 + rr), 7) * pow((-bump2 + rr), 7) +
                        720 * pow((-bump1 + rr), 9) * pow((-bump2 + rr), 7) +
                        1200 * pow((-bump1 + rr), 3) * pow((-bump2 + rr), 9) +
                        3600 * pow((-bump1 + rr), 5) * pow((-bump2 + rr), 9) +
                        720 * pow((-bump1 + rr), 7) * pow((-bump2 + rr), 9) +
                        720 * pow((-bump1 + rr), 5) * pow((-bump2 + rr), 11)))/(pow((bump1 - rr), 12) * pow((bump2 - rr), 12));

}
else {
		f0 = 0.0 ;
		df0 = 0.0 ;
		ddf0 = 0.0;
		d3f0 = 0.0;
		d4f0 = 0.0;
		d5f0 = 0.0;
		d6f0 = 0.0;
	}
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//Setting up the radial functions

if (multipole_l==2) { 
	if ( rr > 1.0e-7 ) {
		A_lm = ( amp_teuk * 0.5 * sqrt(21.0) 
			* ( ( a_0 + b_0 ) * ddf0 / pow(rr,3)
		    	+ ( a_0 - b_0 ) * ddf0 / pow(rr,3)
		    	- 3.0 * ( a_0 + b_0 ) * df0 / pow(rr,4)
		    	- 3.0 * ( a_0 - b_0 ) * df0 / pow(rr,4)
		    	+ 3.0 * ( a_0 + b_0 ) * f0 / pow(rr,5)
		    	+ 3.0 * ( a_0 - b_0 ) * f0 / pow(rr,5)
	 	 	)) ;

		B_lm = (amp_teuk * 0.5 * sqrt(7.0/12.0)
			* ( ( a_0 + b_0 ) * d3f0 / pow(rr,2)
		    	+ ( a_0 - b_0 ) * d3f0 / pow(rr,2)
 		    	- 3.0 * ( a_0 + b_0 ) * ddf0 / pow(rr,3)
		    	- 3.0 * ( a_0 - b_0 ) * ddf0 / pow(rr,3)
		    	+ 6.0 * ( a_0 + b_0 ) * df0 / pow(rr,4)
		    	+ 6.0 * ( a_0 - b_0 ) * df0 / pow(rr,4)
		    	- 6.0 * ( a_0 + b_0 ) * f0 / pow(rr,5)
		    	- 6.0 * ( a_0 - b_0 ) * f0 / pow(rr,5)
			));

		C_lm = (amp_teuk * 0.5 * 0.25 * sqrt(7.0/3.0)
			* ( ( a_0 + b_0 ) * d4f0 / rr
		    	+ ( a_0 - b_0 ) * d4f0 / rr
		    	- 2.0 * ( a_0 + b_0 ) * d3f0 / pow(rr,2)
		    	- 2.0 * ( a_0 - b_0 ) * d3f0 / pow(rr,2)
		    	+ 3.0 * ( a_0 + b_0 ) * ddf0 / pow(rr,3)
		    	+ 3.0 * ( a_0 - b_0 ) * ddf0 / pow(rr,3)
		    	- 3.0 * ( a_0 + b_0 ) * df0 / pow(rr,4)
		    	- 3.0 * ( a_0 - b_0 ) * df0 / pow(rr,4)
		    	+ 3.0 * ( a_0 + b_0 ) * f0 / pow(rr,5)
		    	+ 3.0 * ( a_0 - b_0 ) * f0 / pow(rr,5)
			));

		K_lm = 0.0;
		
		L_lm = 0.0;
	} else if ( rr < 1.0e-7 ) {
		A_lm = (amp_teuk * 0.5 * sqrt(21.0)
			* ( ( a_0 + b_0 ) * ddf0 / pow(r_origin,3)
		    	+ ( a_0 - b_0 ) * ddf0 / pow(r_origin,3)
		    	- 3.0 * ( a_0 + b_0 ) * df0 / pow(r_origin,4)
		    	- 3.0 * ( a_0 - b_0 ) * df0 / pow(r_origin,4)
		    	+ 3.0 * ( a_0 + b_0 ) * f0 / pow(r_origin,5)
		    	+ 3.0 * ( a_0 - b_0 ) * f0 / pow(r_origin,5)
			));

		B_lm = (amp_teuk * 0.5 * sqrt(7.0/12.0)
			* ( ( a_0 + b_0 ) * d3f0 / pow(r_origin,2)
			    + ( a_0 - b_0 ) * d3f0 / pow(r_origin,2)
			    - 3.0 * ( a_0 + b_0 ) * ddf0 / pow(r_origin,3)
			    - 3.0 * ( a_0 - b_0 ) * ddf0 / pow(r_origin,3)
			    + 6.0 * ( a_0 + b_0 ) * df0 / pow(r_origin,4)
			    + 6.0 * ( a_0 - b_0 ) * df0 / pow(r_origin,4)
			    - 6.0 * ( a_0 + b_0 ) * f0 / pow(r_origin,5)
			    - 6.0 * ( a_0 - b_0 ) * f0 / pow(r_origin,5)
			));

		C_lm = (amp_teuk * 0.5 * 0.25 * sqrt(7.0/3.0)
			* (         ( a_0 + b_0 ) * d4f0 / r_origin
			    +       ( a_0 - b_0 ) * d4f0 / r_origin
			    - 2.0 * ( a_0 + b_0 ) * d3f0 / pow(r_origin,2)
			    - 2.0 * ( a_0 - b_0 ) * d3f0 / pow(r_origin,2)
			    + 3.0 * ( a_0 + b_0 ) * ddf0 / pow(r_origin,3)
			    + 3.0 * ( a_0 - b_0 ) * ddf0 / pow(r_origin,3)
			    - 3.0 * ( a_0 + b_0 ) * df0 / pow(r_origin,4)
			    - 3.0 * ( a_0 - b_0 ) * df0 / pow(r_origin,4)
			    + 3.0 * ( a_0 + b_0 ) * f0 / pow(r_origin,5)
			    + 3.0 * ( a_0 - b_0 ) * f0 / pow(r_origin,5)
			));

		K_lm = 0.0;
		
		L_lm = 0.0;
           }
} else if (multipole_l==3) {
			
	if ( rr > 1.0e-7 ) {
		A_lm = (amp_teuk * 0.5 * 3 * sqrt(10) 
			* (          ( a_0 + b_0 ) * d3f0 / pow(rr,3)
		    	    +        ( a_0 - b_0 ) * d3f0 / pow(rr,3)
		    	    -  6.0 * ( a_0 + b_0 ) * ddf0 / pow(rr,4)
		    	    -  6.0 * ( a_0 - b_0 ) * ddf0 / pow(rr,4)
		    	    + 15.0 * ( a_0 + b_0 ) * df0 / pow(rr,5)
		    	    + 15.0 * ( a_0 - b_0 ) * df0 / pow(rr,5)
			    - 15.0 * ( a_0 + b_0 ) * f0 / pow(rr,6)
			    - 15.0 * ( a_0 - b_0 ) * f0 / pow(rr,6)
	 	 	));

		B_lm = (amp_teuk * 0.5 * sqrt(5.0/8.0)
			* (         ( a_0 + b_0 ) * d4f0 / pow(rr,2)
		    	   +        ( a_0 - b_0 ) * d4f0 / pow(rr,2)
 		    	   -  6.0 * ( a_0 + b_0 ) * d3f0 / pow(rr,3)
		    	   -  6.0 * ( a_0 - b_0 ) * d3f0 / pow(rr,3)
		    	   + 21.0 * ( a_0 + b_0 ) * ddf0 / pow(rr,4)
		    	   + 21.0 * ( a_0 - b_0 ) * ddf0 / pow(rr,4)
		    	   - 45.0 * ( a_0 + b_0 ) * df0 / pow(rr,5)
		    	   - 45.0 * ( a_0 - b_0 ) * df0 / pow(rr,5)
			   + 45.0 * ( a_0 + b_0 ) * f0 / pow(rr,6)
			   + 45.0 * ( a_0 - b_0 ) * f0 / pow(rr,6)
			));

		C_lm = (amp_teuk * 0.5 * 0.5 * sqrt(1.0/10.0)
			* (         ( a_0 + b_0 ) * d5f0 / rr
		    	   +        ( a_0 - b_0 ) * d5f0 / rr
		    	   -  5.0 * ( a_0 + b_0 ) * d4f0 / pow(rr,2)
		    	   -  5.0 * ( a_0 - b_0 ) * d4f0 / pow(rr,2)
		    	   + 15.0 * ( a_0 + b_0 ) * d3f0 / pow(rr,3)
		    	   + 15.0 * ( a_0 - b_0 ) * d3f0 / pow(rr,3)
		    	   - 30.0 * ( a_0 + b_0 ) * ddf0 / pow(rr,4)
		    	   - 30.0 * ( a_0 - b_0 ) * ddf0 / pow(rr,4)
		    	   + 45.0 * ( a_0 + b_0 ) * df0 / pow(rr,5)
		    	   + 45.0 * ( a_0 - b_0 ) * df0 / pow(rr,5)
			   - 45.0 * ( a_0 + b_0 ) * f0 / pow(rr,6)
			   - 45.0 * ( a_0 - b_0 ) * f0 / pow(rr,6)
			));

		K_lm = 0.0;
		
		L_lm = 0.0;
	} else if ( rr < 1.0e-7 ) {
		A_lm = (amp_teuk * 0.5 * 3 * sqrt(10) 
			* (          ( a_0 + b_0 ) * d3f0 / pow(r_origin,3)
		    	    +        ( a_0 - b_0 ) * d3f0 / pow(r_origin,3)
		    	    -  6.0 * ( a_0 + b_0 ) * ddf0 / pow(r_origin,4)
		    	    -  6.0 * ( a_0 - b_0 ) * ddf0 / pow(r_origin,4)
		    	    + 15.0 * ( a_0 + b_0 ) * df0 / pow(r_origin,5)
		    	    + 15.0 * ( a_0 - b_0 ) * df0 / pow(r_origin,5)
			    - 15.0 * ( a_0 + b_0 ) * f0 / pow(r_origin,6)
			    - 15.0 * ( a_0 - b_0 ) * f0 / pow(r_origin,6)
	 	 	));

		B_lm = (amp_teuk * 0.5 * sqrt(5.0/8.0)
			* (         ( a_0 + b_0 ) * d4f0 / pow(r_origin,2)
		    	   +        ( a_0 - b_0 ) * d4f0 / pow(r_origin,2)
 		    	   -  6.0 * ( a_0 + b_0 ) * d3f0 / pow(r_origin,3)
		    	   -  6.0 * ( a_0 - b_0 ) * d3f0 / pow(r_origin,3)
		    	   + 21.0 * ( a_0 + b_0 ) * ddf0 / pow(r_origin,4)
		    	   + 21.0 * ( a_0 - b_0 ) * ddf0 / pow(r_origin,4)
		    	   - 45.0 * ( a_0 + b_0 ) * df0 / pow(r_origin,5)
		    	   - 45.0 * ( a_0 - b_0 ) * df0 / pow(r_origin,5)
			   + 45.0 * ( a_0 + b_0 ) * f0 / pow(r_origin,6)
			   + 45.0 * ( a_0 - b_0 ) * f0 / pow(r_origin,6)
			));

		C_lm = (amp_teuk * 0.5 * 0.5 * sqrt(1.0/10.0)
			* (         ( a_0 + b_0 ) * d5f0 / r_origin
		    	   +        ( a_0 - b_0 ) * d5f0 / r_origin
		    	   -  5.0 * ( a_0 + b_0 ) * d4f0 / pow(r_origin,2)
		    	   -  5.0 * ( a_0 - b_0 ) * d4f0 / pow(r_origin,2)
		    	   + 15.0 * ( a_0 + b_0 ) * d3f0 / pow(r_origin,3)
		    	   + 15.0 * ( a_0 - b_0 ) * d3f0 / pow(r_origin,3)
		    	   - 30.0 * ( a_0 + b_0 ) * ddf0 / pow(r_origin,4)
		    	   - 30.0 * ( a_0 - b_0 ) * ddf0 / pow(r_origin,4)
		    	   + 45.0 * ( a_0 + b_0 ) * df0 / pow(r_origin,5)
		    	   + 45.0 * ( a_0 - b_0 ) * df0 / pow(r_origin,5)
			   - 45.0 * ( a_0 + b_0 ) * f0 / pow(r_origin,6)
			   - 45.0 * ( a_0 - b_0 ) * f0 / pow(r_origin,6)
			));

		K_lm = 0.0;
		
		L_lm = 0.0;
           }
}  else if (multipole_l==4) {
			
	if ( rr > 1.0e-7 ) {
		A_lm = (amp_teuk * 0.5 * sqrt(770.0/3.0)  
			* (           ( a_0 + b_0 ) * d4f0 / pow(rr,3)
		    	    +         ( a_0 - b_0 ) * d4f0 / pow(rr,3)
		    	    -  10.0 * ( a_0 + b_0 ) * d3f0 / pow(rr,4)
		    	    -  10.0 * ( a_0 - b_0 ) * d3f0 / pow(rr,4)
		    	    +  45.0 * ( a_0 + b_0 ) * ddf0 / pow(rr,5)
		    	    +  45.0 * ( a_0 - b_0 ) * ddf0 / pow(rr,5)
			    - 105.0 * ( a_0 + b_0 ) * df0 / pow(rr,6)
			    - 105.0 * ( a_0 - b_0 ) * df0 / pow(rr,6)
			    + 105.0 * ( a_0 + b_0 ) * f0 / pow(rr,7)
			    + 105.0 * ( a_0 - b_0 ) * f0 / pow(rr,7)
	 	 	));

		B_lm = (amp_teuk * 0.5 * sqrt(77.0/120.0)
			* (          ( a_0 + b_0 ) * d5f0 / pow(rr,2)
		    	   +         ( a_0 - b_0 ) * d5f0 / pow(rr,2)
 		    	   -  10.0 * ( a_0 + b_0 ) * d4f0 / pow(rr,3)
		    	   -  10.0 * ( a_0 - b_0 ) * d4f0 / pow(rr,3)
		    	   +  55.0 * ( a_0 + b_0 ) * d3f0 / pow(rr,4)
		    	   +  55.0 * ( a_0 - b_0 ) * d3f0 / pow(rr,4)
		    	   - 195.0 * ( a_0 + b_0 ) * ddf0 / pow(rr,5)
		    	   - 195.0 * ( a_0 - b_0 ) * ddf0 / pow(rr,5)
			   + 420.0 * ( a_0 + b_0 ) * df0 / pow(rr,6)
			   + 420.0 * ( a_0 - b_0 ) * df0 / pow(rr,6)
			   - 420.0 * ( a_0 + b_0 ) * f0 / pow(rr,7)
			   - 420.0 * ( a_0 - b_0 ) * f0 / pow(rr,7)
			));

		C_lm = (amp_teuk * 0.5 * ( 1.0/18.0 ) * sqrt(77.0/30.0)
			* (         ( a_0 + b_0 ) * d6f0 / rr
		    	   +        ( a_0 - b_0 ) * d6f0 / rr
		    	   -   9.0 * ( a_0 + b_0 ) * d5f0 / pow(rr,2)
		    	   -   9.0 * ( a_0 - b_0 ) * d5f0 / pow(rr,2)
		    	   +  45.0 * ( a_0 + b_0 ) * d4f0 / pow(rr,3)
		    	   +  45.0 * ( a_0 - b_0 ) * d4f0 / pow(rr,3)
		    	   - 150.0 * ( a_0 + b_0 ) * d3f0 / pow(rr,4)
		    	   - 150.0 * ( a_0 - b_0 ) * d3f0 / pow(rr,4)
		    	   + 360.0 * ( a_0 + b_0 ) * ddf0 / pow(rr,5)
		    	   + 360.0 * ( a_0 - b_0 ) * ddf0 / pow(rr,5)
			   - 630.0 * ( a_0 + b_0 ) * f0 / pow(rr,6)
			   - 630.0 * ( a_0 - b_0 ) * f0 / pow(rr,6)
			   + 630.0 * ( a_0 + b_0 ) * f0 / pow(rr,7)
			   + 640.0 * ( a_0 - b_0 ) * f0 / pow(rr,7)
			));

		K_lm = 0.0;
		
		L_lm = 0.0;
	} else if ( rr < 1.0e-7 ) {
		A_lm = (amp_teuk * 0.5 * sqrt(770.0/3.0)  
			* (           ( a_0 + b_0 ) * d4f0 / pow(r_origin,3)
		    	    +         ( a_0 - b_0 ) * d4f0 / pow(r_origin,3)
		    	    -  10.0 * ( a_0 + b_0 ) * d3f0 / pow(r_origin,4)
		    	    -  10.0 * ( a_0 - b_0 ) * d3f0 / pow(r_origin,4)
		    	    +  45.0 * ( a_0 + b_0 ) * ddf0 / pow(r_origin,5)
		    	    +  45.0 * ( a_0 - b_0 ) * ddf0 / pow(r_origin,5)
			    - 105.0 * ( a_0 + b_0 ) * df0 / pow(r_origin,6)
			    - 105.0 * ( a_0 - b_0 ) * df0 / pow(r_origin,6)
			    + 105.0 * ( a_0 + b_0 ) * f0 / pow(r_origin,7)
			    + 105.0 * ( a_0 - b_0 ) * f0 / pow(r_origin,7)
	 	 	));

		B_lm = (amp_teuk * 0.5 * sqrt(77.0/120.0)
			* (          ( a_0 + b_0 ) * d5f0 / pow(r_origin,2)
		    	   +         ( a_0 - b_0 ) * d5f0 / pow(r_origin,2)
 		    	   -  10.0 * ( a_0 + b_0 ) * d4f0 / pow(r_origin,3)
		    	   -  10.0 * ( a_0 - b_0 ) * d4f0 / pow(r_origin,3)
		    	   +  55.0 * ( a_0 + b_0 ) * d3f0 / pow(r_origin,4)
		    	   +  55.0 * ( a_0 - b_0 ) * d3f0 / pow(r_origin,4)
		    	   - 195.0 * ( a_0 + b_0 ) * ddf0 / pow(r_origin,5)
		    	   - 195.0 * ( a_0 - b_0 ) * ddf0 / pow(r_origin,5)
			   + 420.0 * ( a_0 + b_0 ) * df0 / pow(r_origin,6)
			   + 420.0 * ( a_0 - b_0 ) * df0 / pow(r_origin,6)
			   - 420.0 * ( a_0 + b_0 ) * f0 / pow(r_origin,7)
			   - 420.0 * ( a_0 - b_0 ) * f0 / pow(r_origin,7)
			));

		C_lm = (amp_teuk * 0.5 * ( 1.0/18.0 ) * sqrt(77.0/30.0)
			* (         ( a_0 + b_0 ) * d6f0 / r_origin
		    	   +        ( a_0 - b_0 ) * d6f0 / r_origin
		    	   -   9.0 * ( a_0 + b_0 ) * d5f0 / pow(r_origin,2)
		    	   -   9.0 * ( a_0 - b_0 ) * d5f0 / pow(r_origin,2)
		    	   +  45.0 * ( a_0 + b_0 ) * d4f0 / pow(r_origin,3)
		    	   +  45.0 * ( a_0 - b_0 ) * d4f0 / pow(r_origin,3)
		    	   - 150.0 * ( a_0 + b_0 ) * d3f0 / pow(r_origin,4)
		    	   - 150.0 * ( a_0 - b_0 ) * d3f0 / pow(r_origin,4)
		    	   + 360.0 * ( a_0 + b_0 ) * ddf0 / pow(r_origin,5)
		    	   + 360.0 * ( a_0 - b_0 ) * ddf0 / pow(r_origin,5)
			   - 630.0 * ( a_0 + b_0 ) * f0 / pow(r_origin,6)
			   - 630.0 * ( a_0 - b_0 ) * f0 / pow(r_origin,6)
			   + 630.0 * ( a_0 + b_0 ) * f0 / pow(r_origin,7)
			   + 640.0 * ( a_0 - b_0 ) * f0 / pow(r_origin,7)
			));

		K_lm = 0.0;
		
		L_lm = 0.0;
        }
}



// Now for the angular functions for l=2 
///////////////////////////////////////////////////////////////////////////
// Starting with Y_20
//
Y20  = ( 1.0/4.0 ) * sqrt(5.0/pi) * ( 3.0 * pow(cos_theta,2) - 1.0 );
Y21  = ( 1.0/2.0 ) * sqrt(15.0/pi) * sin_theta * cos_theta * cos_phi;
Y2n1 = ( 1.0/2.0 ) * sqrt(15.0/pi) * sin_theta * cos_theta * sin_phi;
Y22  = ( 1.0/4.0 ) * sqrt(15.0/pi) * pow(sin_theta,2) * cos_2phi;
Y2n2 = ( 1.0/4.0 ) * sqrt(15.0/pi) * pow(sin_theta,2) * sin_2phi;

// Y_2m,theta
//
Y20_theta  = - ( 3.0/2.0 ) * sqrt(5.0/pi) * sin_theta * cos_theta ;
Y21_theta  =   ( 1.0/2.0 ) * sqrt( 15.0/pi) * cos_2theta * cos_phi;
Y2n1_theta =   ( 1.0/2.0 ) * sqrt( 15.0/pi) * cos_2theta * sin_phi;
Y22_theta  =   ( 1.0/2.0 ) * sqrt( 15.0/pi) * sin_theta * cos_theta * cos_2phi;
Y2n2_theta =   ( 1.0/2.0 ) * sqrt( 15.0/pi) * sin_theta * cos_theta * sin_2phi;

// Y_2m,phi
//
Y20_phi  =   0.0;
Y21_phi  = - ( 1.0/2.0 ) * sqrt(15.0/pi) * cos_theta * sin_phi;
Y2n1_phi =   ( 1.0/2.0 ) * sqrt(15.0/pi) * cos_theta * cos_phi;
Y22_phi  = - ( 1.0/2.0 ) * sqrt(15.0/pi) * sin_theta * sin_2phi;
Y2n2_phi =   ( 1.0/2.0 ) * sqrt(15.0/pi) * sin_theta * cos_2phi;

// Y_2m,theta,theta + .5 l ( l + 1 ) Y_2m
//
Y20_thth  =   ( 3.0/4.0 ) * sqrt(5.0/pi) * pow(sin_theta,2);
Y21_thth  = - ( 1.0/2.0 ) * sqrt(15.0/pi) * sin_theta * cos_theta * cos_phi;
Y2n1_thth = - ( 1.0/2.0 ) * sqrt(15.0/pi) * sin_theta * cos_theta * sin_phi;
Y22_thth =    ( 1.0/8.0 ) * sqrt(15.0/pi) * ( 3.0 + cos_2theta ) * cos_2phi;
Y2n2_thth =   ( 1.0/8.0 ) * sqrt(15.0/pi) * ( 3.0 + cos_2theta ) * sin_2phi;

// Y_2m,theta,phi - cot_theta Y-2m,phi
//
Y20_thphi  =   0.0;
Y21_thphi  =   ( 1.0/2.0 ) * sqrt(15.0/pi) * sin_theta * sin_phi;
Y2n1_thphi = - ( 1.0/2.0 ) * sqrt(15.0/pi) * sin_theta * cos_phi;
Y22_thphi  = - ( 1.0/2.0 ) * sqrt(15.0/pi) * cos_theta * sin_2phi;
Y2n2_thphi =   ( 1.0/2.0 ) * sqrt(15.0/pi) * cos_theta * cos_2phi;
      
//l=3 Angular Functions
/////////////////////////////////////////////////////////////////////
//
// Y_3m
Y30  = ( 1.0/8.0 ) * sqrt(7.0/pi) * cos_theta * ( 5.0 * cos_2theta - 1.0 );
Y31  = ( 1.0/16.0 ) * sqrt(21.0/2.0/pi) * ( sin_theta + 5.0 * sin_3theta ) * cos_phi;
Y3n1 = ( 1.0/16.0 ) * sqrt(21.0/2.0/pi) * ( sin_theta + 5.0 * sin_3theta ) * sin_phi;
Y32  = ( 1.0/4.0 ) * sqrt(105.0/pi) * pow(sin_theta,2) * cos_theta * cos_2phi;
Y3n2 = ( 1.0/4.0 ) * sqrt(105.0/pi) * pow(sin_theta,2) * cos_theta * sin_2phi;
Y33  = ( 1.0/4.0 ) * sqrt(35.0/2.0/pi) * pow(sin_theta,3) * cos_3phi;
Y3n3 = ( 1.0/4.0 ) * sqrt(35.0/2.0/pi) * pow(sin_theta,3) * sin_3phi;

// Y_3m,theta
//
Y30_theta  = - ( 3.0/8.0 ) * sqrt(7.0/pi) * ( 3.0 + 5.0 * cos_2theta ) * sin_theta;
Y31_theta  =   ( 1.0/16.0 ) * sqrt(21.0/2.0/pi) * (cos_theta + 15.0 * cos_3theta ) * cos_phi;
Y3n1_theta =   ( 1.0/16.0 ) * sqrt(21.0/2.0/pi) * (cos_theta + 15.0 * cos_3theta ) * sin_phi;
Y32_theta  =   ( 1.0/8.0 ) * sqrt(105.0/pi) * sin_theta * ( 1.0 + 3.0 * cos_2theta ) * cos_2phi;
Y3n2_theta =   ( 1.0/8.0 ) * sqrt(105.0/pi) * sin_theta * ( 1.0 + 3.0 * cos_2theta ) * sin_2phi;
Y33_theta  =   ( 3.0/4.0 ) * sqrt(35.0/2.0/pi) * cos_theta * pow(sin_theta,2) * cos_3phi;
Y3n3_theta =   ( 3.0/4.0 ) * sqrt(35.0/2.0/pi) * cos_theta * pow(sin_theta,2) * sin_3phi;

// Y_3m,phi
//
Y30_phi  =   0.0;
Y31_phi  = - ( 1.0/8.0 ) * sqrt(21.0/2.0/pi) * ( 3.0 + 5.0 * cos_2theta ) * sin_phi;
Y3n1_phi =   ( 1.0/8.0 ) * sqrt(21.0/2.0/pi) * ( 3.0 + 5.0 * cos_2theta ) * cos_phi;
Y32_phi  = - ( 1.0/4.0 ) * sqrt(105.0/pi) * sin_2theta * sin_2phi;
Y3n2_phi =   ( 1.0/4.0 ) * sqrt(105.0/pi) * sin_2theta * cos_2phi;
Y33_phi  = - ( 3.0/4.0 ) * sqrt(35.0/2.0/pi) * pow(sin_theta,2) * sin_3phi;
Y3n3_phi =   ( 3.0/4.0 ) * sqrt(35.0/2.0/pi) * pow(sin_theta,2) * cos_3phi;

// Y_3m,theta,theta + .5 l ( l+1 ) Y_3m
// 
Y30_thth  = ( 15.0/4.0 ) * sqrt(7.0/pi) * cos_theta * pow(sin_theta,2);
Y31_thth  = ( 5.0/16.0 ) * sqrt(21.0/2.0/pi) * ( sin_theta - 3.0 * sin_3theta ) * cos_phi;
Y3n1_thth = ( 5.0/16.0 ) * sqrt(21.0/2.0/pi) * ( sin_theta - 3.0 * sin_3theta ) * sin_phi;
Y32_thth  = ( 1.0/16.0 ) * sqrt(105.0/pi) * ( 5.0 * cos_theta + 3.0 * cos_3theta ) * cos_2phi;
Y3n2_thth = ( 1.0/16.0 ) * sqrt(105.0/pi) * ( 5.0 * cos_theta + 3.0 * cos_3theta ) * sin_2phi;
Y33_thth  = ( 3.0/16.0 ) * sqrt(35.0/2.0/pi) * ( 5.0 * sin_theta + sin_3theta ) * cos_3phi;
Y3n3_thth  = ( 3.0/16.0 ) * sqrt(35.0/2.0/pi) * ( 5.0 * sin_theta + sin_3theta ) * sin_3phi;

// Y_3m,theta,phi - cot_theta Y_3m_phi
//
Y30_thphi  =   0.0;
Y31_thphi  =   ( 5.0/4.0 ) * sqrt(21.0/2.0/pi) * sin_2theta * sin_phi;
Y3n1_thphi = - ( 5.0/4.0 ) * sqrt(21.0/2.0/pi) * sin_2theta * cos_phi;
Y32_thphi  = - ( 1.0/2.0 ) * sqrt(105.0/pi) * cos_2theta * sin_2phi;
Y3n2_thphi =   ( 1.0/2.0 ) * sqrt(105.0/pi) * cos_2theta * cos_2phi;
Y33_thphi  = - ( 3.0/4.0 ) * sqrt(35.0/2.0/pi) * sin_2theta * sin_3phi;
Y3n3_thphi =   ( 3.0/4.0 ) * sqrt(35.0/2.0/pi) * sin_2theta * cos_3phi;  


// l=4 angular functions
/////////////////////////////////////////////////////////////////////////////////////////
//
//Y_4m
//
Y40  = ( 3.0/128.0 ) * sqrt( 1.0/pi ) * (9.0 + 20.0 * cos_2theta + 35.0 * cos_4theta);
Y41  = ( 3.0/32.0 ) * sqrt( 5.0/2.0/pi ) * ( 2.0 * sin_2theta + 7.0 * sin_4theta ) * cos_phi;
Y4n1 = ( 3.0/32.0 ) * sqrt( 5.0/2.0/pi ) * ( 2.0 * sin_2theta + 7.0 * sin_4theta ) * sin_phi;
Y42  = ( 3.0/16.0 ) * sqrt( 5.0/pi ) * ( 5.0 + 7.0 * cos_2theta ) * pow(sin_theta,2) * cos_2phi;
Y4n2 = ( 3.0/16.0 ) * sqrt( 5.0/pi ) * ( 5.0 + 7.0 * cos_2theta ) * pow(sin_theta,2) * sin_2phi;
Y43  = ( 3.0/4.0 ) * sqrt( 35.0/2/pi ) * cos_theta * pow(sin_theta,3) * cos_3phi;
Y4n3 = ( 3.0/4.0 ) * sqrt( 35.0/2/pi ) * cos_theta * pow(sin_theta,3) * sin_3phi;
Y44  = ( 3.0/16.0 ) * sqrt( 35.0/pi ) * pow(sin_theta,4) * cos_4phi;
Y4n4 = ( 3.0/16.0 ) * sqrt( 35.0/pi ) * pow(sin_theta,4) * sin_4phi;

//Y_4m,theta
//
Y40_theta  = ( 15.0/4.0 ) * sqrt( 1.0/pi ) * ( 3.0 - 7.0 * pow(cos_theta,2) ) * sin_theta;
Y41_theta  = ( 3.0/8.0 ) * sqrt( 5.0/2.0/pi ) * (cos_2theta + 7.0 * cos_4theta) * cos_phi;
Y4n1_theta = ( 3.0/8.0 ) * sqrt( 5.0/2.0/pi ) * (cos_2theta + 7.0 * cos_4theta) * sin_phi;
Y42_theta  = ( 3.0/16.0 ) * sqrt( 5.0/pi ) * ( -2.0 * sin_2theta + 7.0 * sin_4theta ) * cos_2phi;
Y4n2_theta = ( 3.0/16.0 ) * sqrt( 5.0/pi ) * ( -2.0 * sin_2theta + 7.0 * sin_4theta ) * sin_2phi;
Y43_theta  = ( 3.0/4.0 ) * sqrt( 35.0/2.0/pi ) * (1.0 + 2.0 * cos_2theta) * pow(sin_theta,2) * cos_3phi; 
Y4n3_theta = ( 3.0/4.0 ) * sqrt( 35.0/2.0/pi ) * (1.0 + 2.0 * cos_2theta) * pow(sin_theta,2) * sin_3phi;
Y44_theta  = ( 3.0/4.0 ) * sqrt( 35.0/pi ) * cos_theta * pow(sin_theta,3) * cos_4phi;
Y4n4_theta = ( 3.0/4.0 ) * sqrt( 35.0/pi ) * cos_theta * pow(sin_theta,3) * sin_4phi;

//Y_4m,phi
//
Y40_phi  =   0.0;
Y41_phi  = - ( 3.0/16.0 ) * sqrt( 5.0/2.0/pi ) * ( 9.0 * cos_theta + 7.0 * cos_3theta ) * sin_phi;
Y4n1_phi =   ( 3.0/16.0 ) * sqrt( 5.0/2.0/pi ) * ( 9.0 * cos_theta + 7.0 * cos_3theta ) * cos_phi;
Y42_phi  = - ( 3.0/8.0 ) * sqrt( 5.0/pi ) * ( 5.0 + 7.0 * cos_2theta ) * sin_theta * sin_2phi;
Y4n2_phi =   ( 3.0/8.0 ) * sqrt( 5.0/pi ) * ( 5.0 + 7.0 * cos_2theta ) * sin_theta * cos_2phi;
Y43_phi  = - ( 9.0/4.0 ) * sqrt( 35.0/2.0/pi ) * cos_theta * pow(sin_theta,2) * sin_3phi;
Y4n3_phi =   ( 9.0/4.0 ) * sqrt( 35.0/2.0/pi ) * cos_theta * pow(sin_theta,2) * cos_3phi;
Y44_phi  = - ( 3.0/4.0 ) * sqrt( 35.0/pi ) * pow(sin_theta,3) * sin_4phi;
Y4n4_phi =   ( 3.0/4.0 ) * sqrt( 35.0/pi ) * pow(sin_theta,3) * cos_4phi;

//Y_4m,theta,theta + .5 l ( l+1 ) Y_4m
//
Y40_thth  = ( 45.0/16.0 ) * sqrt( 1.0/pi ) * ( 5.0 + 7.0 * cos_2theta ) * pow(sin_theta,2);
Y41_thth  = ( 9.0/16.0 ) * sqrt( 5.0/2.0/pi ) * ( 2.0 * sin_2theta - 7.0 * sin_4theta ) * cos_phi;
Y4n1_thth = ( 9.0/16.0 ) * sqrt( 5.0/2.0/pi ) * ( 2.0 * sin_2theta - 7.0 * sin_4theta ) * sin_phi;
Y42_thth  = ( 9.0/32.0 ) * sqrt( 5.0/pi ) * ( 5.0 + 4.0 * cos_2theta + 7.0 * cos_4theta ) * cos_2phi;
Y4n2_thth = ( 9.0/32.0 ) * sqrt( 5.0/pi ) * ( 5.0 + 4.0 * cos_2theta + 7.0 * cos_4theta ) * sin_2phi;
Y43_thth  = ( 9.0/2.0 ) * sqrt( 35.0/2.0/pi ) * pow(cos_theta,3) * sin_theta * cos_3phi;
Y4n3_thth = ( 9.0/2.0 ) * sqrt( 35.0/2.0/pi ) * pow(cos_theta,3) * sin_theta * sin_3phi;
Y44_thth  = ( 9.0/16.0 ) * sqrt( 35.0/pi ) * ( 3.0 + cos_2theta ) * pow(sin_theta,2) * cos_4phi;
Y4n4_thth = ( 9.0/16.0 ) * sqrt( 35.0/pi ) * ( 3.0 + cos_2theta ) * pow(sin_theta,2) * sin_4phi;

//Y_4m,theta,phi - cot_theta Y_4m,phi
//
Y40_thphi  =   0.0;
Y41_thphi  =   ( 9.0/8.0 ) * sqrt( 5.0/2.0/pi ) * ( 5.0 + 7.0 * cos_2theta ) * sin_theta * sin_phi;
Y4n1_thphi = - ( 9.0/8.0 ) * sqrt( 5.0/2.0/pi ) * ( 5.0 + 7.0 * cos_2theta ) * sin_theta * cos_phi;
Y42_thphi  = - ( 9.0/16.0 ) * sqrt( 5.0/pi ) * ( cos_theta + 7.0 * cos_3theta ) * sin_2phi;
Y4n2_thphi =   ( 9.0/16.0 ) * sqrt( 5.0/pi ) * ( cos_theta + 7.0 * cos_3theta ) * cos_2phi;
Y43_thphi  = - ( 9.0/8.0 ) * sqrt( 35.0/2.0/pi ) * (1.0 + 3.0 * cos_2theta ) * sin_theta * sin_3phi;
Y4n3_thphi =   ( 9.0/8.0 ) * sqrt( 35.0/2.0/pi ) * (1.0 + 3.0 * cos_2theta ) * sin_theta * cos_3phi;
Y44_thphi  = - ( 9.0/4.0 ) * sqrt( 35.0/pi ) * cos_theta * pow(sin_theta,2) * sin_4phi;
Y4n4_thphi =   ( 9.0/4.0 ) * sqrt( 35.0/pi ) * cos_theta * pow(sin_theta,2) * cos_4phi;

////////////////////////////////////////////////////////////////////////////////////////
if (multipole_l == 2) {
	if (multipole_m == 0) {
		Ylm = Y20;
		Ylm_theta = Y20_theta;
		Ylm_thth = Y20_thth;
		Ylm_thphi = Y20_thphi;
		Ylm_phi = Y20_phi;
	} else if (multipole_m == 1) {
		Ylm = Y21;
		Ylm_theta = Y21_theta;
		Ylm_thth = Y21_thth;
		Ylm_thphi = Y21_thphi;
		Ylm_phi = Y21_phi;	
	} else if (multipole_m == -1) {
		Ylm = Y2n1;
		Ylm_theta = Y2n1_theta;
		Ylm_thth = Y2n1_thth;
		Ylm_thphi = Y2n1_thphi;
		Ylm_phi = Y2n1_phi;	
	} else if (multipole_m == 2) {
		Ylm = Y22;
		Ylm_theta = Y22_theta;
		Ylm_thth = Y22_thth;
		Ylm_thphi = Y22_thphi;
		Ylm_phi = Y22_phi;
	} else if (multipole_m == -2) {
		Ylm = Y2n2;
		Ylm_theta = Y2n2_theta;
		Ylm_thth = Y2n2_thth;
		Ylm_thphi = Y2n2_thphi;
		Ylm_phi = Y2n2_phi;
	} else {
		std::cout << "unnallowed m for l = 2/n";
	}

} else if (multipole_l == 3) {
	if (multipole_m == 0){
		Ylm = Y30;
		Ylm_theta = Y30_theta;
		Ylm_thth = Y30_thth;
		Ylm_thphi = Y30_thphi;
		Ylm_phi = Y30_phi;
	} else if (multipole_m == 1) {
		Ylm = Y31;
		Ylm_theta = Y31_theta;
		Ylm_thth = Y31_thth;
		Ylm_thphi = Y31_thphi;
		Ylm_phi = Y31_phi;	
	} else if (multipole_m == -1) {
		Ylm = Y3n1;
		Ylm_theta = Y3n1_theta;
		Ylm_thth = Y3n1_thth;
		Ylm_thphi = Y3n1_thphi;
		Ylm_phi = Y3n1_phi;	
	} else if (multipole_m == 2) {
		Ylm = Y32;
		Ylm_theta = Y32_theta;
		Ylm_thth = Y32_thth;
		Ylm_thphi = Y32_thphi;
		Ylm_phi = Y32_phi;
	} else if (multipole_m == -2) {
		Ylm = Y3n2;
		Ylm_theta = Y3n2_theta;
		Ylm_thth = Y3n2_thth;
		Ylm_thphi = Y3n2_thphi;
		Ylm_phi = Y3n2_phi;
	} else if (multipole_m == 3) {
		Ylm = Y33;
		Ylm_theta = Y33_theta;
		Ylm_thth = Y33_thth;
		Ylm_thphi = Y33_thphi;
		Ylm_phi = Y33_phi;
	} else if (multipole_m == -3) {
		Ylm = Y3n3;
		Ylm_theta = Y3n3_theta;
		Ylm_thth = Y3n3_thth;
		Ylm_thphi = Y3n3_thphi;
		Ylm_phi = Y3n3_phi;
 
	} else {
		std::cout << "unnallowed m for l = 3/n";
	}


} else if (multipole_l == 4) {
	if (multipole_m == 0){
		Ylm = Y40;
		Ylm_theta = Y40_theta;
		Ylm_thth = Y40_thth;
		Ylm_thphi = Y40_thphi;
		Ylm_phi = Y40_phi;
	} else if (multipole_m == 1) {
		Ylm = Y41;
		Ylm_theta = Y41_theta;
		Ylm_thth = Y41_thth;
		Ylm_thphi = Y41_thphi;
		Ylm_phi = Y41_phi;	
	} else if (multipole_m == -1) {
		Ylm = Y4n1;
		Ylm_theta = Y4n1_theta;
		Ylm_thth = Y4n1_thth;
		Ylm_thphi = Y4n1_thphi;
		Ylm_phi = Y4n1_phi;	
	} else if (multipole_m == 2) {
		Ylm = Y42;
		Ylm_theta = Y42_theta;
		Ylm_thth = Y42_thth;
		Ylm_thphi = Y42_thphi;
		Ylm_phi = Y42_phi;
	} else if (multipole_m == -2) {
		Ylm = Y4n2;
		Ylm_theta = Y4n2_theta;
		Ylm_thth = Y4n2_thth;
		Ylm_thphi = Y4n2_thphi;
		Ylm_phi = Y4n2_phi;
	} else if (multipole_m == 3) {
		Ylm = Y43;
		Ylm_theta = Y43_theta;
		Ylm_thth = Y43_thth;
		Ylm_thphi = Y43_thphi;
		Ylm_phi = Y43_phi;
	} else if (multipole_m == -3) {
		Ylm = Y4n3;
		Ylm_theta = Y4n3_theta;
		Ylm_thth = Y4n3_thth;
		Ylm_thphi = Y4n3_thphi;
		Ylm_phi = Y4n3_phi;
	} else if (multipole_m == 4) {
		Ylm = Y44;
		Ylm_theta = Y44_theta;
		Ylm_thth = Y44_thth;
		Ylm_thphi = Y44_thphi;
		Ylm_phi = Y44_phi;
	} else if (multipole_m == -4) {
		Ylm = Y4n4;
		Ylm_theta = Y4n4_theta;
		Ylm_thth = Y4n4_thth;
		Ylm_thphi = Y4n4_thphi;
		Ylm_phi = Y4n4_phi;

	} else {
		std::cout << "unnallowed m for l = 4/n";
	}
}

////////////////////////////////////////////////////////////////////////////////
//Now put it all together to get the perturbed metric
h_bare_rr   = A_lm * Ylm;
h_bare_rth  = B_lm * Ylm_theta - K_lm * Ylm_phi;
h_bare_rph  = B_lm * Ylm_phi + K_lm * Ylm_theta;
h_bare_thth =  - 0.5 * A_lm * Ylm + C_lm * Ylm_thth - L_lm * Ylm_thphi;
h_bare_thph = C_lm * Ylm_thphi + L_lm * Ylm_thth;
h_bare_phph = -(  0.5 * A_lm * Ylm + C_lm * Ylm_thth - L_lm * Ylm_thphi);

g_bare_rr   = 1.0 + h_bare_rr;
g_bare_rth  =        h_bare_rth;
g_bare_rph  =        h_bare_rph;
g_bare_thth = 1.0 + h_bare_thth;
g_bare_thph =        h_bare_thph;
g_bare_phph = 1.0 + h_bare_phph;

g_rr   =                           g_bare_rr;
g_rth  = rr                       * g_bare_rth;
g_rph  = rr * sin_theta           * g_bare_rph;
g_thth = rr*rr                     * g_bare_thth;
g_thph = rr*rr*sin_theta           * g_bare_thph;
g_phph = rr*rr*sin_theta*sin_theta * g_bare_phph;



// We now have to transform our metric from spherical coordinates
// to Cartesian coordinates.  But we have to pay particular attention
// to the axis and the origin of coordinates.
if ( rho > 1.0e-7 ) { // everywhere but the axis ...
	drdx = xx / rr;
	drdy = yy / rr;
	drdz = zz / rr;

	dthdx = xx * zz / ( rr*rr*rho );
	dthdy = yy * zz / ( rr*rr*rho );
	dthdz = - rho /( rr*rr );

	dphdx = - yy / ( rho_sqrd );
	dphdy =   xx / ( rho_sqrd );
	dphdz = 0.0;

	g_xx =  (  drdx *  drdx * g_rr
		+ dthdx * dthdx * g_thth
		+ dphdx * dphdx * g_phph
		+ 2.0 * (    drdx * dthdx * g_rth
			+  drdx * dphdx * g_rph
			+ dthdx * dphdx * g_thph  ));
	g_xy =  (  drdx *  drdy * g_rr
		+ dthdx * dthdy * g_thth
		+ dphdx * dphdy * g_phph
		+ (  drdx*dthdy +  drdy*dthdx ) * g_rth
		+ (  drdx*dphdy +  drdy*dphdx ) * g_rph
		+ ( dthdx*dphdy + dthdy*dphdx ) * g_thph);
	g_xz =  ( drdx *  drdz * g_rr
		+ dthdx * dthdz * g_thth
		+ dphdx * dphdz * g_phph
		+ (  drdx*dthdz +  drdz*dthdx ) * g_rth
		+ (  drdx*dphdz +  drdz*dphdx ) * g_rph
		+ ( dthdx*dphdz + dthdz*dphdx ) * g_thph);
	g_yy =  (drdy *  drdy * g_rr
		+ dthdy * dthdy * g_thth
		+ dphdy * dphdy * g_phph
		+ 2.0 * (    drdy * dthdy * g_rth
			+  drdy * dphdy * g_rph
			+ dthdy * dphdy * g_thph  ));
	g_yz =  (  drdy *  drdz * g_rr
		+ dthdy * dthdz * g_thth
		+ dphdy * dphdz * g_phph
		+ (  drdy*dthdz +  drdz*dthdy ) * g_rth
		+ (  drdy*dphdz +  drdz*dphdy ) * g_rph
		+ ( dthdy*dphdz + dthdz*dphdy ) * g_thph);
	g_zz =  (  drdz *  drdz * g_rr
		+ dthdz * dthdz * g_thth
		+ dphdz * dphdz * g_phph
		+ 2.0 * (    drdz * dthdz * g_rth
			+  drdz * dphdz * g_rph
			+ dthdz * dphdz * g_thph  )); 

} else if ( rho < 1.0e-7 && rr > 1.0e-7 ) { // on axis but not origin

	rho_axis = sqrt( rho_sqrd + epsilon_sq );

	g_xx =  (  g_bare_rr
		- ( yy*yy + zz*zz ) * ( g_bare_rr - g_bare_thth ) / (rr*rr)
		- yy*yy * ( g_bare_thth - g_bare_phph ) / (rho_axis*rho_axis)
		+ 2.0/rr * (xx/rho_axis)
			* ( xx*zz/rr * g_bare_rth
			    - yy * g_bare_rph
			    - zz*(yy/rho_axis) * g_thph  ));
	g_xy =  ( xx*yy/(rr*rr) * ( g_bare_rr - g_bare_phph )
		+ xx*yy/(rho_axis*rho_axis) * zz*zz/(rr*rr)
			* ( g_bare_thth - g_bare_phph )
		+ 2.0 * zz/(rr*rr) * (xx/rho_axis) * yy * g_bare_rth
		+ (xx/rho_axis) * xx * g_bare_rph / rr
		- (yy/rho_axis) * yy * g_bare_rph / rr
		+ pow((xx/rho_axis),2) * zz * g_bare_thph / rr
		- pow((yy/rho_axis),2) * zz * g_bare_thph / rr);
	g_xz =  (  xx*zz/(rr*rr) * ( g_bare_rr - g_bare_thth )
		+ (   zz*zz * (xx/rho_axis) * g_bare_rth
		- xx*rho * g_bare_rth ) / (rr*rr)
		- (zz/rr) * (yy/rho_axis) * g_bare_rph
		+ yy * g_bare_thph / rr) ;
	g_yy =  (  g_bare_rr
		- (xx*xx + zz*zz)/(rr*rr) * ( g_bare_rr - g_bare_thth )
		- pow((xx/rho_axis),2) * ( g_bare_thth - g_bare_phph )
		+ 2.0/rr * (yy/rho_axis)
			* ( yy*zz/rr * g_bare_rth
			    + (xx/rho_axis)*zz * g_bare_thph
			    + xx * g_bare_rph ));
	g_yz =  (  yy*zz/(rr*rr) * ( g_bare_rr - g_bare_thth )
		+ (yy/rho_axis)*zz*zz/(rr*rr) * g_bare_rth
		- yy*rho/(rr*rr) * g_bare_rth
		+ (zz/rr) * (xx/rho_axis) * g_bare_rph
		- xx * g_bare_thph / rr);
	g_zz = (  g_bare_rr
              - (rho*rho)/(rr*rr) * ( g_bare_rr - g_bare_thth )
              - 2.0 * zz * rho * g_bare_rth / (rr*rr));

} else if ( rho < 1.0e-7 && rr < 1.0e-7) { // at the origin

	rho_axis = sqrt( rho_sqrd + epsilon_sq );
	r_origin = sqrt( r_sqrd + epsilon_sq );


	g_xx =  (  g_bare_rr
		- ( yy*yy + zz*zz ) / (pow(r_origin,2))
			* ( g_bare_rr - g_bare_thth )
		- yy*yy / (rho_axis*rho_axis)
			* ( g_bare_thth - g_bare_phph )
		+ 2.0/(r_origin) * (xx/rho_axis)
			* ( xx*zz/r_origin * g_bare_rth
			   - yy * g_bare_rph
			   - zz*(yy/rho_axis) * g_thph  ));
	g_xy =  ( xx*yy/(pow(r_origin,2)) * ( g_bare_rr - g_bare_phph )
		+ xx*yy/(rho_axis*rho_axis) * zz*zz/(pow(r_origin,2))
			* ( g_bare_thth - g_bare_phph )
		+ 2.0 * zz/(pow(r_origin,2)) * (xx/rho_axis) * yy * g_bare_rth
		+ (xx/rho_axis) * (xx/r_origin) * g_bare_rph
		- (yy/rho_axis) * (yy/r_origin) * g_bare_rph
		+ pow((xx/rho_axis),2) * (zz/r_origin) * g_bare_thph
		- pow((yy/rho_axis),2) * (zz/r_origin) * g_bare_thph);
	g_xz =  ( xx*zz/(pow(r_origin,2)) * ( g_bare_rr - g_bare_thth )
		+ (   pow((zz/r_origin),2) * (xx/rho_axis) * g_bare_rth
		- (xx/r_origin)*(rho/r_origin) * g_bare_rth )
		- (zz/r_origin) * (yy/rho_axis) * g_bare_rph
		+ (yy/r_origin) * g_bare_thph);
	g_yy =  ( g_bare_rr
		- ( pow((xx/r_origin),2) + pow((zz/r_origin),2))
			* ( g_bare_rr - g_bare_thth )
		- pow((xx/rho_axis),2) * ( g_bare_thth - g_bare_phph )
		+ 2.0 * (yy/rho_axis)
			* (  (yy/r_origin)*(zz/r_origin) * g_bare_rth
			   + (xx/rho_axis)*(zz/r_origin) * g_bare_thph
			   + (xx/r_origin) * g_bare_rph ));
	g_yz =  (  (yy/r_origin)*(zz/r_origin) * ( g_bare_rr - g_bare_thth )
		+ (yy/rho_axis)*pow((zz/r_origin),2) * g_bare_rth
		- (yy/r_origin)*(rho/r_origin) * g_bare_rth
		+ (zz/r_origin) * (xx/rho_axis) * g_bare_rph
		- (xx/r_origin) * g_bare_thph);
	g_zz =  (  g_bare_rr
		- pow((rho/r_origin),2) * ( g_bare_rr - g_bare_thth )
		- 2.0 * (zz/r_origin) * (rho/r_origin) * g_bare_rth);

} else {
	std::cout << " something strange happened ... ";
}
       
//Checking to see if the determinant is negative
determinant = (g_xx*g_yy*g_zz)+(g_xy*g_yz*g_xz)+(g_xz*g_xy*g_yz)-((g_xz*g_yy*g_xz)+(g_yz*g_yz*g_xx)+(g_zz*g_xy*g_xy));

if (determinant<0) {
	std::cout << "The determinant = " <<determinant<<std::endl;
	std::cout <<" r = "<< rr<<std::endl;
	std::cout <<" x = "<< xx<<std::endl;
	std::cout <<" y = "<< yy<<std::endl;
	std::cout <<" z = " <<zz<<std::endl;
	
	//Other debugging options
	/*
	std::cout <<" g_xx =" << g_xx << std::endl;
    std::cout <<" g_xy =" << g_xy << std::endl;
    std::cout <<" g_xz =" << g_xz << std::endl;
    std::cout <<" g_yy =" << g_yy << std::endl;
    std::cout <<" g_yz =" << g_yz << std::endl;
    std::cout <<" g_zz =" << g_zz << std::endl;
	*/

	/*
	std::cout <<" g_bare_rr = " << g_bare_rr << std::endl;    
	std::cout <<" g_bare_rth = " << g_bare_rth << std::endl; 
	std::cout <<" g_bare_rph = " << g_bare_rph << std::endl;  
	std::cout <<" g_bare_thth = " << g_bare_thth << std::endl;    
	std::cout <<" g_bare_thph = " << g_bare_thph << std::endl;    
	std::cout <<" g_bare_phph = " << g_bare_phph << std::endl;   
    */

 	/*  
	std::cout <<" h_bare_rr = " << h_bare_rr << std::endl;    
	std::cout <<" h_bare_rth = " << h_bare_rth << std::endl; 
	std::cout <<" h_bare_rph = " << h_bare_rph << std::endl;  
	std::cout <<" h_bare_thth = " << h_bare_thth << std::endl;    
	std::cout <<" h_bare_thph = " << h_bare_thph << std::endl;    
	std::cout <<" h_bare_phph = " << h_bare_phph << std::endl; 
	*/

	/*
	std::cout <<" grid_spacing = " << grid_spacing;
	std::cout <<" rho_axis = " << rho_axis<<std::endl;
	std::cout <<" r_origin = " << r_origin<<std::endl;
	std::cout <<" epsilon_sq = " << epsilon_sq<<std::endl;
	*/

	/*
	std::cout <<" A_lm = " << A_lm << std::endl;
	std::cout <<" B_lm = " << B_lm << std::endl;
	std::cout <<" C_lm = " << C_lm << std::endl;
	std::cout <<" K_lm = " << K_lm << std::endl;
	std::cout <<" L_lm = " << L_lm << std::endl;
	*/
}
var[VAR::U_ALPHA]  = 1.0;
var[VAR::U_BETA0] = 0.0;
var[VAR::U_BETA1] = 0.0;
var[VAR::U_BETA2] = 0.0;

var[VAR::U_B0] = 0.0;
var[VAR::U_B1] = 0.0;
var[VAR::U_B2] = 0.0;

var[VAR::U_GT0] = 0.0;
var[VAR::U_GT1] = 0.0;
var[VAR::U_GT2] = 0.0;

var[VAR::U_SYMGT0] = g_xx;
var[VAR::U_SYMGT1] = g_xy;
var[VAR::U_SYMGT2] = g_xz;
var[VAR::U_SYMGT3] = g_yy;
var[VAR::U_SYMGT4] = g_yz;
var[VAR::U_SYMGT5] = g_zz;

//	var[VAR::U_G00] = g_xx;
//	var[VAR::U_G01] = g_xy;
//	var[VAR::U_G02] = g_xz;
//	var[VAR::U_G11] = g_yy;
//	var[VAR::U_G12] = g_yz;
//	var[VAR::U_G22] = g_zz;

var[VAR::U_SYMAT0] =  0.0;
var[VAR::U_SYMAT1] =  0.0;
var[VAR::U_SYMAT2] =  0.0;
var[VAR::U_SYMAT3] =  0.0;
var[VAR::U_SYMAT4] =  0.0;
var[VAR::U_SYMAT5] =  0.0;

var[VAR::U_CHI]   = 1.0;
var[VAR::U_K]   = 0.0;
//	var[VAR::U_THETA] = 0.0;

//	var[VAR::U_FLAG] = 0.0;
//	var[VAR::U_TUU00] = 0.0;
//	var[VAR::U_TUU01] = 0.0;
//	var[VAR::U_TUU02] = 0.0;
//	var[VAR::U_TUU03] = 0.0;
//	var[VAR::U_TUU11] = 0.0;
//	var[VAR::U_TUU12] = 0.0;
//	var[VAR::U_TUU13] = 0.0;
//	var[VAR::U_TUU22] = 0.0;
//	var[VAR::U_TUU23] = 0.0;
//	var[VAR::U_TUU33] = 0.0;
