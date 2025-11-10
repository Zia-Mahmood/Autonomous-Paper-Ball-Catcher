/*
  4WD Mecanum Wheel Robot Car Base - Motor Test
  4wd-test.ino
  Cycles all 4 Mecanum wheels to verify proper operation
  Uses ESP32 DevKitC (other ESP32 boards will work)
  Uses 2 TB6612FNG Motor Drivers, also compatible with L298N
  
  DroneBot Workshop 2022
  https://dronebotworkshop.com
*/
 
// Define Motor Connections
// Right Front Motor
#define MF_PWMA 19
#define MF_AI1 32
#define MF_AI2 23
 
// Left Front Motor
#define MF_PWMB 26
#define MF_BI1 33
#define MF_BI2 25
 
// Right Rear Motor
#define MR_PWMA 27
#define MR_AI1 12
#define MR_AI2 14
 
// Left Rear Motor
#define MR_PWMB 4
#define MR_BI1 13
#define MR_BI2 2
 
// Define some preset motor speeds (0 - 255, adjust as desired)
int speed_slow = 100;
int speed_fast = 250;
 
void setup() {
 
  // Set all connections as outputs
  pinMode(MF_PWMA, OUTPUT);
  pinMode(MF_AI1, OUTPUT);
  pinMode(MF_AI2, OUTPUT);
  pinMode(MF_PWMB, OUTPUT);
  pinMode(MF_BI1, OUTPUT);
  pinMode(MF_BI2, OUTPUT);
  pinMode(MR_PWMA, OUTPUT);
  pinMode(MR_AI1, OUTPUT);
  pinMode(MR_AI2, OUTPUT);
  pinMode(MR_PWMB, OUTPUT);
  pinMode(MR_BI1, OUTPUT);
  pinMode(MR_BI2, OUTPUT);
}
 
void loop() {
 
  // Front Right Motor  *****************************************************************
 
  // FR - Forward Slow Speed
  digitalWrite(MF_AI1, HIGH);
  digitalWrite(MF_AI2, LOW);
  analogWrite(MF_PWMA, speed_slow);
  delay(2000);
 
  // FR - Forward Fast Speed
  analogWrite(MF_PWMA, speed_fast);
  delay(2000);
 
  // FR - Stop 1 second
  analogWrite(MF_PWMA, 0);
  delay(1000);
 
  // FR - Reverse Fast Speed
  digitalWrite(MF_AI1, LOW);
  digitalWrite(MF_AI2, HIGH);
  analogWrite(MF_PWMA, speed_fast);
  delay(2000);
 
  // FR - Reverse Slow Speed
  analogWrite(MF_PWMA, speed_slow);
  delay(2000);
 
  // FR - Stop 1 second
  analogWrite(MF_PWMA, 0);
  delay(1000);
 
  // Front Left Motor  *****************************************************************
 
  // FL - Forward Slow Speed
  digitalWrite(MF_BI1, HIGH);
  digitalWrite(MF_BI2, LOW);
  analogWrite(MF_PWMB, speed_slow);
  delay(2000);
 
  // FL - Forward Fast Speed
  analogWrite(MF_PWMB, speed_fast);
  delay(2000);
 
  // FL - Stop 1 second
  analogWrite(MF_PWMB, 0);
  delay(1000);
 
  // FL - Reverse Fast Speed
  digitalWrite(MF_BI1, LOW);
  digitalWrite(MF_BI2, HIGH);
  analogWrite(MF_PWMB, speed_fast);
  delay(2000);
 
  // FL - Reverse Slow Speed
  analogWrite(MF_PWMB, speed_slow);
  delay(2000);
 
  // FL - Stop 1 second
  analogWrite(MF_PWMB, 0);
  delay(1000);
 
  // Rear Right Motor  *****************************************************************
 
  // RR - Forward Slow Speed
  digitalWrite(MR_AI1, HIGH);
  digitalWrite(MR_AI2, LOW);
  analogWrite(MR_PWMA, speed_slow);
  delay(2000);
 
  // RR - Forward Fast Speed
  analogWrite(MR_PWMA, speed_fast);
  delay(2000);
 
  // RR - Stop 1 second
  analogWrite(MR_PWMA, 0);
  delay(1000);
 
  // RR - Reverse Fast Speed
  digitalWrite(MR_AI1, LOW);
  digitalWrite(MR_AI2, HIGH);
  analogWrite(MR_PWMA, speed_fast);
  delay(2000);
 
  // RR - Reverse Slow Speed
  analogWrite(MR_PWMA, speed_slow);
  delay(2000);
 
  // RR - Stop 1 second
  analogWrite(MR_PWMA, 0);
  delay(1000);
 
  // Rear Left Motor  *****************************************************************
 
  // RL - Forward Slow Speed
  digitalWrite(MR_BI1, HIGH);
  digitalWrite(MR_BI2, LOW);
  analogWrite(MR_PWMB, speed_slow);
  delay(2000);
 
  // RL - Forward Fast Speed
  analogWrite(MR_PWMB, speed_fast);
  delay(2000);
 
  // RL - Stop 1 second
  analogWrite(MR_PWMB, 0);
  delay(1000);
 
  // RL - Reverse Fast Speed
  digitalWrite(MR_BI1, LOW);
  digitalWrite(MR_BI2, HIGH);
  analogWrite(MR_PWMB, speed_fast);
  delay(2000);
 
  // RL - Reverse Slow Speed
  analogWrite(MR_PWMB, speed_slow);
  delay(2000);
 
  // RL - Stop 1 second
  analogWrite(MR_PWMB, 0);
  delay(1000);
}