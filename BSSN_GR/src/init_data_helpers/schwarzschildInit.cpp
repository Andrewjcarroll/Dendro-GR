double m = 5.0 ; 

//double o4=x*x;
//double o5=y*y;
//double o6=z*z;
double o4=xx*xx;
double o5=yy*yy;
double o6=zz*zz;
double o7=o4+o5+o6;
double o8;
       
       if ( fabs(xx) > 1.0e-4 || fabs(yy) > 1.0e-4 || fabs(zz) > 1.0e-4 )
       { o8=sqrt(o7); } 
       else 
       //{ o8=1.e-6; } 
       { o8=1.e-10; } 

double o3=0.5*m;
double o9=o3+o8;
double o21;
       
       if ( fabs(xx) > 1.0e-4 || fabs(yy) > 1.0e-4 ) 
       { o21=o4+o5; } 
       else 
       //{ o21 = 1.e-6; } 
       { o21 = 1.e-10; } 
       
double o22=pow(o21,-2.);
double o23=-2.*o22*o4;
double o24=-2.*o22*o5;
double o25=1/o21;
double o26=2.*o25;
double o27=o23+o24+o26;
     var[VAR::U_ALPHA]=sqrt(pow(o9,-4.)*(-2.*m*o8*(o9*o9)+pow(o9,4.)));
//var[VAR::U_ALPHA]=(o8-o3)/(o8+o3); 
//var[VAR::U_ALPHA]=(o8-o3)/(o9); 
//var[VAR::U_ALPHA]=sqrt((o9*o9-2*o8*m)/(o9*o9));
//var[VAR::U_ALPHA]=pow((o9*o9-2*o8*m)/(o9*o9), 0.5);
//var[VAR::U_ALPHA]=pow(1.0-2.0*o8*m/(o9*o9), 0.5);
//var[VAR::U_ALPHA]=sqrt((o8-o3)*(o8-o3)/(o9*o9)) ; 
var[VAR::U_CHI]=o7*o7*pow(pow(o9,12.),-0.3333333333333333);
var[VAR::U_K]=0.;
var[VAR::U_GT0]=-(o27*xx);
var[VAR::U_GT1]=-(o27*yy);
//var[VAR::U_CAP_GT0]=0.0; 
//var[VAR::U_CAP_GT1]=0.0; 
var[VAR::U_GT2]=0.;
var[VAR::U_BETA0]=0.;
var[VAR::U_BETA1]=0.;
var[VAR::U_BETA2]=0.;
var[VAR::U_B0]=0.;
var[VAR::U_B1]=0.;
var[VAR::U_B2]=0.;
var[VAR::U_SYMGT0]=1.;
var[VAR::U_SYMGT1]=0.;
var[VAR::U_SYMGT2]=0.;
var[VAR::U_SYMGT3]=1.;
var[VAR::U_SYMGT4]=0.;
var[VAR::U_SYMGT5]=1.;
var[VAR::U_SYMAT0]=0.;
var[VAR::U_SYMAT1]=0.;
var[VAR::U_SYMAT2]=0.;
var[VAR::U_SYMAT3]=0.;
var[VAR::U_SYMAT4]=0.;
var[VAR::U_SYMAT5]=0.;



