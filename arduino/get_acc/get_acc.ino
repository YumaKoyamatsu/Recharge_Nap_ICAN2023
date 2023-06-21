
int Vdd = 3.3 * 1000 ;               //KXR94-2050の電源電圧(mV)を入力 
int offset_voltage = Vdd / 2 ;       //Vdd時の0Gにおけるオフセット電圧(mV)
double sensitivity = Vdd / 5 ;       //1Gあたりの出力振幅（感度）(mV)
float ms2 = 9.80665;                 // 地球の重力である1Gの加速度(m/s^2)
double xyz;

void setup() {
  pinMode(A0,INPUT);
  pinMode(A1,INPUT);
  pinMode(A2,INPUT);
  Serial.begin(9600);
  TCB2.CCMP = 2500;  // TOP値の設定
  TCB2.CTRLB = (TCB2_CTRLB & 0b10101000) + 0b00000000;  //タイマーのGPIO出力ON、クロックソースを設定
  TCB2.CTRLA = (TCB2_CTRLA & 0b11111000) + 0b00000101;  //カウント周期を設定してカウントスタート
  TCB2.INTCTRL = 1;  //割り込み許可
}

void loop() {

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

ISR(TCB2_INT_vect) {
  xyz = imu();
  Serial.println(xyz);
}
