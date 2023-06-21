#include <IIRFilter.h>
//For IIR filter
const double a1[] = {7.14505684988850143e-01, 1.42901136997770029e+00, 7.14505684988850143e-01};
const double b1[] = {1, -8.38321351401799642e-01, 2.02945553420599084e-01};
const double k1 = 1.27579181551399912e-01;
const double a2[] = {7.14505684988850143e-01, 1.42901136997770029e+00, 7.14505684988850143e-01};
const double b2[] = {1, -1.09363168265562494e+00, 5.69301995625692725e-01};
const double k2 = 1.66433354892582347e-01;

// const double a1[] = {5.99617751998835269e-01, 1.19923550399767054e+00, 5.99617751998835269e-01};
// const double b1[] = {1, -8.95842193671120279e-01, 2.11784839607788627e-01};
// const double k1 = 1.31726689579931805e-01;
// const double a2[] = {5.99617751998835269e-01, 1.19923550399767054e+00, 5.99617751998835269e-01};
// const double b2[] = {1, -1.00161337524486482e+00, 3.54859049780033020e-01};
// const double k2 = 1.47279526564056101e-01;
// const double a3[] = {5.99617751998835269e-01, 1.19923550399767054e+00, 5.99617751998835269e-01};
// const double b3[] = {1, -1.25910150491803297e+00, 7.03157236805880936e-01};
// const double k3 = 1.85141171357744605e-01;

int data = 0;  
int data_iir = 0;    

//IIR filter fourth order
IIRFilter iir_1(a1, b1);  // IIRフィルタ
IIRFilter iir_2(a2, b2);  // IIRフィルタ

void setup(){
    Serial.begin(115200);
}

void loop(){
    //Getting data
    data = analogRead(A3);
    //Filtering
    data_iir = iir_filter(data);
    debug_print();
    delay(10);
}

void debug_print(){
    Serial.print(data);
    Serial.print(",");
    Serial.print(data_iir);
    
    Serial.println("");
}

int iir_filter(int data){
    data_iir = iir_1.filter(data * k1);
    data_iir = iir_2.filter(data_iir * k2);
    return data_iir;
}