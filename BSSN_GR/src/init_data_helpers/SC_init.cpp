// Reading the mass in from the par file
mass = bssn::BSSN_BH1_MASS;
//defining some initial variables
double xbar = x;
double ybar = y;
double zbar = z;

double rho;
double r;

//Setting up a case to test the special cases of the code
if (fabs(xbar) >1.0e-4 ||fabs(ybar) > 1.0e-4||fabs(zbar) >1.04e-4){
rho = sqrt(xbar*xbar + ybar*ybar);
}
else{
    //at the origin
    rho=1.0e-10;
}

//Not on axis and not at the origin
if (fabs(xbar) >1.0e-4 ||fabs(ybar) > 1.0e-4||fabs(zbar) >1.04e-4){
r = sqrt(xbar*xbar + ybar*ybar + zbar*zbar);
}
else{
    //at the origin
    r=1.0e-10;
}


//defining the initial values of the variables the evolution variables:
var[VAR::U_ALPHA] = (1.0-2.0*mass/r)/(1.0+2.0*mass/r);
var[VAR::U_CHI]=;
//We can set the trace of the extrinsic curvature to 0 to start with
var[VAR::U_K]=0.;
//What is this?
var[VAR::U_GT0]=-0;
var[VAR::U_GT1]=0;
var[VAR::U_GT2]=0;
//The shift vector disappears in Schwartzchild
var[VAR::U_BETA0]=0.;
var[VAR::U_BETA1]=0.;
var[VAR::U_BETA2]=0.;
//Gauge
var[VAR::U_B0]=0.;
var[VAR::U_B1]=0.;
var[VAR::U_B2]=0.;
//We can define the metric to be minkowski space everywhere to start with
var[VAR::U_SYMGT0]=1.;
var[VAR::U_SYMGT1]=0.;
var[VAR::U_SYMGT2]=0.;
var[VAR::U_SYMGT3]=1.;
var[VAR::U_SYMGT4]=0.;
var[VAR::U_SYMGT5]=1.;
//AT is the Transverse Traceless 
var[VAR::U_SYMAT0]=0.;
var[VAR::U_SYMAT1]=0.;
var[VAR::U_SYMAT2]=0.;
var[VAR::U_SYMAT3]=0.;
var[VAR::U_SYMAT4]=0.;
var[VAR::U_SYMAT5]=0.;