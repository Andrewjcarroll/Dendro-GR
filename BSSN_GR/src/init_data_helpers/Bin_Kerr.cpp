double Alpha_1;
double Chi_1;
double Trk_1;

double Beta0_1;
double Beta1_1;
double Beta2_1;

double GT0_1;
double GT1_1;
double GT2_1;

double GaugeB0_1;
double GaugeB1_1;
double GaugeB2_1;

double GT00_1;
double GT01_1;
double GT02_1;
double GT11_1;
double GT12_1;
double GT22_1;

double AT00_1;
double AT01_1;
double AT02_1;
double AT11_1;
double AT12_1;
double AT22_1;


//Fields from black hole 2

double Alpha_2;
double Chi_2;
double Trk_2;

double Beta0_2;
double Beta1_2;
double Beta2_2;

double GT0_2;
double GT1_2;
double GT2_2;

double GaugeB0_2;
double GaugeB1_2;
double GaugeB2_2;

double GT00_2;
double GT01_2;
double GT02_2;
double GT11_2;
double GT12_2;
double GT22_2;

double AT00_2;
double AT01_2;
double AT02_2;
double AT11_2;
double AT12_2;
double AT22_2;

double offset = 4.0;
double mass = 0.5 ; 
double a_spin = 0.1 ; 
// double mass = emda::EMDA_SINGLE_BH_MASS ; 
// double a_spin = emda::EMDA_SINGLE_BH_SPIN ; 
double rbar ; 
double cyl_rho_bar ; 
double xbar = xx+ offset; 
double ybar = yy; 
double zbar = zz; 

double cyl_rho_bar_sqr = xbar*xbar + ybar*ybar ; 
double rbar_sqr = cyl_rho_bar_sqr + zbar*zbar ; 

double cos_theta ;
double sin_theta ; 


if ( fabs(xbar) > 1.0e-4 || fabs(ybar) > 1.0e-4 ) // not on axis  
{
    cyl_rho_bar = sqrt( cyl_rho_bar_sqr ) ;
    rbar = sqrt( rbar_sqr ) ;
    cos_theta = zbar / rbar ;
    sin_theta = cyl_rho_bar / rbar ;
}
else // on axis  
{
    cyl_rho_bar = 1.0e-10 ;

    if ( fabs(zbar) > 1.0e-4 ) // on axis but not at the origin  
    {
        rbar = sqrt( rbar_sqr ) ;
        cos_theta = zbar / rbar ;
        sin_theta = 0.0 ; //cyl_rho_bar / rbar ; 
    }
    else // at the origin  
    {
        rbar = 1.0e-10 ;
        cos_theta = 0.0 ;
        sin_theta = 0.0 ;
    }
}


double r_boylin; 
double f_0 ; 
double rho_bar_sqr ; 
double Delta_bar ; 
double Sigma_bar ; 
double CC ; 
double beta_phi_up ; 

f_0 =  (rbar + 0.5*mass)*(rbar + 0.5*mass) - 0.25*a_spin*a_spin ;
rho_bar_sqr = f_0*f_0 + a_spin*a_spin*zbar*zbar ;
Delta_bar = f_0*f_0 - 2*mass*rbar*f_0 + a_spin*a_spin*rbar*rbar ;
Sigma_bar =   pow(f_0*f_0 + a_spin*a_spin*rbar*rbar , 2 ) 
            - a_spin*a_spin*(xbar*xbar+ybar*ybar)*Delta_bar ;
CC = rho_bar_sqr*rho_bar_sqr / Sigma_bar ;
beta_phi_up = - a_spin * 2*mass * f_0 * pow(rbar,3) / Sigma_bar ;

r_boylin = f_0 / rbar ;

Alpha_1= sqrt( Delta_bar * rho_bar_sqr / Sigma_bar ) ;


Chi_1 = pow(rbar, 4)/pow(Sigma_bar*rho_bar_sqr, 1.0/3.0) ;
//var[VAR::U_CHI] = pow(rbar/(rbar+0.5*mass), 4);
Trk_1 = 0.0 ;

Beta0_1 = - ybar * beta_phi_up ;    // shift_x  
Beta1_1 =   xbar * beta_phi_up ;    // shift_y   
Beta2_1 = 0.0 ;    // shift_z 

//var[VAR::U_CAP_GT0] = 0.0 ;  // Gt_x 
//var[VAR::U_CAP_GT1] = 0.0 ;  // Gt_y 
//var[VAR::U_CAP_GT2] = 0.0 ;  // Gt_z 
        
GT0_1 = pow(a_spin,2) * xbar * rho_bar_sqr * pow(CC,-4.0/3.0)/3.0 / pow(Sigma_bar,2) * ( 2 * pow(sin_theta,2) * sqrt( Delta_bar ) * ( rho_bar_sqr * ( f_0 - mass*rbar ) + 2*f_0*rbar*(2*mass*f_0) ) + 2*pow(cos_theta,2) * (rho_bar_sqr*Delta_bar - 2 *Sigma_bar ) - 3*rho_bar_sqr*(rho_bar_sqr + 2*mass*f_0*rbar) ) ;

GT1_1 = pow(a_spin,2) * ybar * rho_bar_sqr * pow(CC,-4.0/3.0)/3.0 / pow(Sigma_bar,2) * ( 2 * pow(sin_theta,2) * sqrt( Delta_bar ) * ( rho_bar_sqr * ( f_0 - mass*rbar ) + 2*f_0*rbar*(2*mass*f_0) ) + 2*pow(cos_theta,2) * (rho_bar_sqr*Delta_bar - 2 *Sigma_bar ) - 3*rho_bar_sqr*(rho_bar_sqr + 2*mass*f_0*rbar) ) ;

GT2_1 = 2*pow(a_spin,2) * zbar * pow(sin_theta,2) * rho_bar_sqr * pow(CC,-4.0/3.0)/3.0 / pow(Sigma_bar,2) * ( sqrt( Delta_bar ) * ( rho_bar_sqr * ( f_0 - mass*rbar ) + 2*f_0*rbar*(2*mass*f_0) ) - (rho_bar_sqr*Delta_bar - 2 *Sigma_bar ) ) ;




//var[VAR::U_GT00] = pow(CC, -2.0/3.0) * (xbar*xbar*CC + ybar*ybar) / cyl_rho_bar ;
GT00_1 = pow(CC, -2.0/3.0) * (CC - (CC-1.0)*ybar*ybar/cyl_rho_bar) ;
//var[VAR::U_GT00] = 1.0 ; 
GT01_1 = xbar*ybar * pow(CC, -2.0/3.0) * (CC - 1.0) / cyl_rho_bar ;
//var[VAR::U_GT01] = 0.0 ; 
GT02_1 = 0.0 ;
//var[VAR::U_GT11] = pow(CC, -2.0/3.0) * (ybar*ybar*CC + xbar*xbar) / cyl_rho_bar ;
GT11_1 = pow(CC, -2.0/3.0) * (CC -(CC-1.0)*xbar*xbar/cyl_rho_bar) ;
//var[VAR::U_GT11] = 1.0 ; 
GT12_1 = 0.0 ;
GT22_1 = pow(CC, 1.0/3.0) ;

AT00_1 = 2*a_spin*xbar*ybar*rbar/rho_bar_sqr/pow(Sigma_bar*rho_bar_sqr,5.0/6.0) * ( mass*rho_bar_sqr*(pow(f_0,2) + pow(a_spin*rbar,2)) - f_0*(2*mass*f_0)*(rho_bar_sqr+pow(f_0,2) + pow(a_spin*rbar,2) ) + pow(a_spin*zbar,2)*sqrt(Delta_bar)*2*mass*f_0 ) ;

AT01_1 = - a_spin*(xbar*xbar - ybar*ybar)*rbar/rho_bar_sqr/pow(Sigma_bar*rho_bar_sqr,5.0/6.0) * ( mass*rho_bar_sqr*(pow(f_0,2) + pow(a_spin*rbar,2)) - f_0*(2*mass*f_0)*(rho_bar_sqr+pow(f_0,2) + pow(a_spin*rbar,2) ) + pow(a_spin*zbar,2)*sqrt(Delta_bar)*2*mass*f_0 ) ;

AT02_1 = a_spin*ybar*zbar*rbar/rho_bar_sqr/pow(Sigma_bar*rho_bar_sqr,5.0/6.0) * ( mass*rho_bar_sqr*(pow(f_0,2) + pow(a_spin*rbar,2)) - f_0*(2*mass*f_0)*(rho_bar_sqr+pow(f_0,2) + pow(a_spin*rbar,2) ) - pow(a_spin,2)*(xbar*xbar+ybar*ybar)*sqrt(Delta_bar)*2*mass*f_0 ) ;

AT11_1 = - var[VAR::U_SYMAT0] ;

AT12_1 = - a_spin*xbar*zbar*rbar/rho_bar_sqr/pow(Sigma_bar*rho_bar_sqr,5.0/6.0) * ( mass*rho_bar_sqr*(pow(f_0,2) + pow(a_spin*rbar,2)) - f_0*(2*mass*f_0)*(rho_bar_sqr+pow(f_0,2) + pow(a_spin*rbar,2) ) - pow(a_spin,2)*(xbar*xbar+ybar*ybar)*sqrt(Delta_bar)*2*mass*f_0 ) ;

AT22_1 = 0.0 ;

 bool foundNaN = false;

    // Check each variable for NaN
    if (std::isnan(rbar)) {
        std::cout << "rbar is NaN." << std::endl;
        foundNaN = true;
    }





double mass_2 = 0.5 ; 
double a_spin_2 = 0.1 ; 
// double mass_2 = emda::EMDA_SINGLE_BH_mass_2 ; 
// double a_spin_2 = emda::EMDA_SINGLE_BH_SPIN ; 
double rbar_2 ; 
double cyl_rho_bar_2 ; 
double xbar_2 = xx-offset; 
double ybar_2 = yy; 
double zbar_2 = zz; 

double cyl_rho_bar_sqr_2 = xbar_2*xbar_2 + ybar_2*ybar_2 ; 
double rbar_sqr_2 = cyl_rho_bar_sqr_2 + zbar_2*zbar_2 ; 

double cos_theta_2 ;
double sin_theta_2 ; 


if ( fabs(xbar_2) > 1.0e-4 || fabs(ybar_2) > 1.0e-4 ) // not on axis  
{
    cyl_rho_bar_2 = sqrt( cyl_rho_bar_sqr_2 ) ;
    rbar_2 = sqrt( rbar_sqr_2 ) ;
    cos_theta_2 = zbar_2 / rbar_2 ;
    sin_theta_2 = cyl_rho_bar_2 / rbar_2 ;
}
else // on axis  
{
    cyl_rho_bar_2 = 1.0e-10 ;

    if ( fabs(zbar_2) > 1.0e-4 ) // on axis but not at the origin  
    {
        rbar_2 = sqrt( rbar_sqr_2 ) ;
        cos_theta_2 = zbar_2 / rbar_2 ;
        sin_theta_2 = 0.0 ; //cyl_rho_bar_2 / rbar_2 ; 
    }
    else // at the origin  
    {
        rbar_2 = 1.0e-10 ;
        cos_theta_2 = 0.0 ;
        sin_theta_2 = 0.0 ;
    }
}


double r_boylin_2; 
double f_0_2 ; 
double rho_bar_sqr_2; 
double Delta_bar_2 ; 
double Sigma_bar_2 ; 
double CC_2 ; 
double beta_phi_up_2 ; 

f_0_2 =  (rbar_2 + 0.5*mass_2)*(rbar_2 + 0.5*mass_2) - 0.25*a_spin_2*a_spin_2 ;
rho_bar_sqr_2= f_0_2*f_0_2 + a_spin_2*a_spin_2*zbar_2*zbar_2 ;
Delta_bar_2 = f_0_2*f_0_2 - 2*mass_2*rbar_2*f_0_2 + a_spin_2*a_spin_2*rbar_2*rbar_2 ;
Sigma_bar_2 =   pow(f_0_2*f_0_2 + a_spin_2*a_spin_2*rbar_2*rbar_2 , 2 ) 
            - a_spin_2*a_spin_2*(xbar_2*xbar_2+ybar_2*ybar_2)*Delta_bar_2 ;
CC_2 = rho_bar_sqr_2*rho_bar_sqr_2/ Sigma_bar_2 ;
beta_phi_up_2 = - a_spin_2 * 2*mass_2 * f_0_2 * pow(rbar_2,3) / Sigma_bar_2 ;

r_boylin_2 = f_0_2 / rbar_2 ;


Alpha_2= sqrt( Delta_bar_2 * rho_bar_sqr_2 / Sigma_bar_2 ) ;
Chi_2 = pow(rbar_2, 4)/pow(Sigma_bar_2*rho_bar_sqr_2, 1.0/3.0) ;
//var[VAR::U_CHI] = pow(rbar/(rbar_2+0.5*mass), 4);
Trk_2 = 0.0 ;

Beta0_2 = - ybar_2 * beta_phi_up_2 ;    // shift_x  
Beta1_2 =   xbar_2 * beta_phi_up_2 ;    // shift_y   
Beta2_2 = 0.0 ;    // shift_z 

//var[VAR::U_CAP_GT0] = 0.0 ;  // Gt_x 
//var[VAR::U_CAP_GT1] = 0.0 ;  // Gt_y 
//var[VAR::U_CAP_GT2] = 0.0 ;  // Gt_z 
        
GT0_2 = pow(a_spin_2,2) * xbar_2 * rho_bar_sqr_2 * pow(CC_2,-4.0/3.0)/3.0 / pow(Sigma_bar_2,2) * ( 2 * pow(sin_theta_2,2) * sqrt( Delta_bar_2 ) * ( rho_bar_sqr_2 * ( f_0_2 - mass_2*rbar_2 ) + 2*f_0_2*rbar_2*(2*mass_2*f_0_2) ) + 2*pow(cos_theta_2,2) * (rho_bar_sqr_2*Delta_bar_2 - 2 *Sigma_bar_2 ) - 3*rho_bar_sqr_2*(rho_bar_sqr_2 + 2*mass_2*f_0_2*rbar_2) ) ;

GT1_2 = pow(a_spin_2,2) * ybar_2 * rho_bar_sqr_2 * pow(CC_2,-4.0/3.0)/3.0 / pow(Sigma_bar_2,2) * ( 2 * pow(sin_theta_2,2) * sqrt( Delta_bar_2 ) * ( rho_bar_sqr_2 * ( f_0_2 - mass_2*rbar_2 ) + 2*f_0_2*rbar_2*(2*mass_2*f_0_2) ) + 2*pow(cos_theta_2,2) * (rho_bar_sqr_2*Delta_bar_2 - 2 *Sigma_bar_2 ) - 3*rho_bar_sqr_2*(rho_bar_sqr_2 + 2*mass_2*f_0_2*rbar_2) ) ;

GT2_2 = 2*pow(a_spin_2,2) * zbar_2 * pow(sin_theta_2,2) * rho_bar_sqr_2 * pow(CC_2,-4.0/3.0)/3.0 / pow(Sigma_bar_2,2) * ( sqrt( Delta_bar_2 ) * ( rho_bar_sqr_2 * ( f_0_2 - mass_2*rbar_2 ) + 2*f_0_2*rbar_2*(2*mass_2*f_0_2) ) - (rho_bar_sqr_2*Delta_bar_2 - 2 *Sigma_bar_2 ) ) ;




//var[VAR::U_GT00] = pow(CC_2, -2.0/3.0) * (xbar_2*xbar_2*CC_2 + ybar_2*ybar_2) / cyl_rho_bar ;
GT00_2 = pow(CC_2, -2.0/3.0) * (CC_2 - (CC_2-1.0)*ybar_2*ybar_2/cyl_rho_bar_2) ;
//var[VAR::U_GT00] = 1.0 ; 
GT01_2 = xbar_2*ybar_2 * pow(CC_2, -2.0/3.0) * (CC_2 - 1.0) / cyl_rho_bar_2 ;
//var[VAR::U_GT01] = 0.0 ; 
GT02_2 = 0.0 ;
//var[VAR::U_GT11] = pow(CC_2, -2.0/3.0) * (ybar_2*ybar_2*CC_2 + xbar_2*xbar_2) / cyl_rho_bar_2 ;
GT11_2 = pow(CC_2, -2.0/3.0) * (CC_2 -(CC_2-1.0)*xbar_2*xbar_2/cyl_rho_bar_2) ;
//var[VAR::U_GT11] = 1.0 ; 
GT12_2 = 0.0 ;
GT22_2 = pow(CC_2, 1.0/3.0) ;

AT00_2 = 2*a_spin_2*xbar_2*ybar_2*rbar_2/rho_bar_sqr_2/pow(Sigma_bar_2*rho_bar_sqr_2,5.0/6.0) * ( mass_2*rho_bar_sqr_2*(pow(f_0_2,2) + pow(a_spin_2*rbar_2,2)) - f_0_2*(2*mass_2*f_0_2)*(rho_bar_sqr_2+pow(f_0_2,2) + pow(a_spin_2*rbar_2,2) ) + pow(a_spin_2*zbar_2,2)*sqrt(Delta_bar_2)*2*mass_2*f_0_2 ) ;

AT01_2 = - a_spin_2*(xbar_2*xbar_2 - ybar_2*ybar_2)*rbar_2/rho_bar_sqr_2/pow(Sigma_bar_2*rho_bar_sqr_2,5.0/6.0) * ( mass_2*rho_bar_sqr_2*(pow(f_0_2,2) + pow(a_spin_2*rbar_2,2)) - f_0_2*(2*mass_2*f_0_2)*(rho_bar_sqr_2+pow(f_0_2,2) + pow(a_spin_2*rbar_2,2) ) + pow(a_spin_2*zbar_2,2)*sqrt(Delta_bar_2)*2*mass_2*f_0_2 ) ;

AT02_2 = a_spin_2*ybar_2*zbar_2*rbar_2/rho_bar_sqr_2/pow(Sigma_bar_2*rho_bar_sqr_2,5.0/6.0) * ( mass_2*rho_bar_sqr_2*(pow(f_0_2,2) + pow(a_spin_2*rbar_2,2)) - f_0_2*(2*mass_2*f_0_2)*(rho_bar_sqr_2+pow(f_0_2,2) + pow(a_spin_2*rbar_2,2) ) - pow(a_spin_2,2)*(xbar_2*xbar_2+ybar_2*ybar_2)*sqrt(Delta_bar_2)*2*mass_2*f_0_2 ) ;

AT11_2 = - var[VAR::U_SYMAT0] ;

AT12_2 = - a_spin_2*xbar_2*zbar_2*rbar_2/rho_bar_sqr_2/pow(Sigma_bar_2*rho_bar_sqr_2,5.0/6.0) * ( mass_2*rho_bar_sqr_2*(pow(f_0_2,2) + pow(a_spin_2*rbar_2,2)) - f_0_2*(2*mass_2*f_0_2)*(rho_bar_sqr_2+pow(f_0_2,2) + pow(a_spin_2*rbar_2,2) ) - pow(a_spin_2,2)*(xbar_2*xbar_2+ybar_2*ybar_2)*sqrt(Delta_bar_2)*2*mass_2*f_0_2 ) ;

AT22_2 = 0.0 ;

var[VAR::U_ALPHA] = Alpha_1+Alpha_2 -1;
var[VAR::U_CHI] = Chi_1+Chi_2-1;
//var[VAR::U_CHI] = pow(rbar_2_2/(rbar_2_2+0.5*mass_2_2), 4);
var[VAR::U_K] = 0.0 ;

var[VAR::U_BETA0] = 0.5*(Beta0_1+Beta0_2) ;    // shift_x  
var[VAR::U_BETA1] =  0.5*(Beta1_1+Beta1_2) ;    // shift_y   
var[VAR::U_BETA2] = 0.0 ;    // shift_z 


        
var[VAR::U_GT0] = 0.5*(GT0_1+GT0_2);

var[VAR::U_GT1] = 0.5*(GT1_1+GT1_2);

var[VAR::U_GT2] = 0.5*(GT2_1+GT2_2) ;


var[VAR::U_B0] = 0.0 ;  // gaugeB0
var[VAR::U_B1] = 0.0 ;  // gaugeB1
var[VAR::U_B2] = 0.0 ;  // gaugeB2


//var[VAR::U_GT00] = pow(CC_2_2, -2.0/3.0) * (xbar_2_2*xbar_2_2*CC_2_2 + ybar_2_2*ybar_2_2) / cyl_rho_bar_2_2 ;
var[VAR::U_SYMGT0] = GT00_1+GT00_2 -1 ;
//var[VAR::U_GT00] = 1.0 ; 
var[VAR::U_SYMGT1] =  GT01_1+GT01_2;
//var[VAR::U_GT01] = 0.0 ; 
var[VAR::U_SYMGT2] = 0.0 ;
//var[VAR::U_GT11] = pow(CC_2_2, -2.0/3.0) * (ybar_2_2*ybar_2_2*CC_2_2 + xbar_2_2*xbar_2_2) / cyl_rho_bar_2_2 ;
var[VAR::U_SYMGT3] = GT11_1+ GT11_2 -1 ;
//var[VAR::U_GT11] = 1.0 ; 
var[VAR::U_SYMGT4] = 0.0 ;
var[VAR::U_SYMGT5] = GT22_1+ GT22_2 -1;

var[VAR::U_SYMAT0] =  AT00_1+AT00_2;

var[VAR::U_SYMAT1] = AT01_1+AT01_2 ;

var[VAR::U_SYMAT2] =  AT02_1+AT02_2;

var[VAR::U_SYMAT3] = - var[VAR::U_SYMAT0] ;

var[VAR::U_SYMAT4] = AT12_1+AT12_2;
var[VAR::U_SYMAT5] = 0.0;

if (std::isnan(Alpha_1)) {
        std::cout << "Alpha_1 is NaN." << std::endl;
        foundNaN = true;
    }

    // Check if Alpha_2 is NaN
    if (std::isnan(Alpha_2)) {
        std::cout << "Alpha_2 is NaN." << std::endl;
        foundNaN = true;
    }

