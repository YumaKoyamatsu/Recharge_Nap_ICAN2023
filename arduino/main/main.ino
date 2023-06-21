#include <IIRFilter.h>

//For IIR filter
const double a1[] = {7.14505684988850143e-01, 1.42901136997770029e+00, 7.14505684988850143e-01};
const double b1[] = {1, -8.38321351401799642e-01, 2.02945553420599084e-01};
const double k1 = 1.27579181551399912e-01;
const double a2[] = {7.14505684988850143e-01, 1.42901136997770029e+00, 7.14505684988850143e-01};
const double b2[] = {1, -1.09363168265562494e+00, 5.69301995625692725e-01};
const double k2 = 1.66433354892582347e-01;

//IIR filter fourth order
IIRFilter iir_1(a1, b1);  // IIRフィルタ
IIRFilter iir_2(a2, b2);  // IIRフィルタ

//Acc sensor
int Vdd = 3.3 * 1000 ;               //KXR94-2050の電源電圧(mV)を入力 
int offset_voltage = Vdd / 2 ;       //Vdd時の0Gにおけるオフセット電圧(mV)
double sensitivity = Vdd / 5 ;       //1Gあたりの出力振幅（感度）(mV)
float ms2 = 9.80665;                 // 地球の重力である1Gの加速度(m/s^2)

//PPG sensor
int ppg_data = 0;
double ppg_data_iir = 0;  
double acc_data = 0.0;

//time
int sample_time = 20;
long pre_time = micros();

void setup(){
    //シリアルポートを開く
    Serial.begin(115200);

    //ピンの設定
    pinMode(A0, INPUT);
    pinMode(A1, INPUT);
    pinMode(A2, INPUT);
    pinMode(A3, INPUT);
    
}

void loop(){
    //Getting data
    ppg_data = analogRead(A3);
    acc_data = imu();
    
    //Filtering
    ppg_data_iir = iir_filter(ppg_data);
    Serial.print(millis());
    Serial.print(",");
    Serial.print(ppg_data);
    Serial.print(",");
    Serial.print(acc_data, 5);
    Serial.print('\n');
    while(millis()-pre_time < sample_time){
        
    }
    pre_time = millis();
}

double imu(){
    // XYZの電圧(mV)を取得する
    double x =  (analogRead(A0) / 1023.0) * 5.0 * 1000.0;
    double y =  (analogRead(A1) / 1023.0) * 5.0 * 1000.0;
    double z =  (analogRead(A2) / 1023.0) * 5.0 * 1000.0;

    // XYZの出力振幅(mV)を求める
    x = x - offset_voltage;
    y = y - offset_voltage;
    z = z - offset_voltage;

    // XYZから重力(G)を求める
    double xg = double(x) / double(sensitivity);
    double yg = double(y) / double(sensitivity);
    double zg = double(z) / double(sensitivity);

    //キャリブレーション
    xg = (xg-0.09)/1.055;
    yg = (yg-0.1333)/1.04;
    zg = (zg-0.1133)/1.05;

    double xx = pow(xg,2);
    double yy = pow(yg,2);
    double zz = pow(zg,2);
    double xyz = pow(xx+yy+zz,0.50);

    return xyz;
}

double iir_filter(double data){
    double data_iir = iir_1.filter(data * k1);
    data_iir = iir_2.filter(data_iir * k2);
    return data_iir;
}
