#include <WiFi.h>
#include "soc/soc.h"
#include "soc/rtc_cntl_reg.h"

// --- CONFIGURATION ---
const char* ssid = "Thousand-sunny";
const char* password = "ZiaBHai123";

WiFiServer server(80);

// --- MOTOR PINS (Matches your setup) ---
// Right Front (FR)
#define MF_PWMA 19
#define MF_AI1 32
#define MF_AI2 23
// Left Front (FL)
#define MF_PWMB 26
#define MF_BI1 33
#define MF_BI2 25
// Right Rear (RR)
#define MR_PWMA 27
#define MR_AI1 12
#define MR_AI2 14
// Left Rear (RL)
#define MR_PWMB 4
#define MR_BI1 13
#define MR_BI2 2

#define LED_PIN 2

unsigned long lastCmdTime = 0;
const int WATCHDOG_TIMEOUT = 500; // Stop if no data for 0.5s

void setup() {
  WRITE_PERI_REG(RTC_CNTL_BROWN_OUT_REG, 0); 
  Serial.begin(115200);

  // Setup Pins
  pinMode(MF_PWMA, OUTPUT); pinMode(MF_AI1, OUTPUT); pinMode(MF_AI2, OUTPUT);
  pinMode(MF_PWMB, OUTPUT); pinMode(MF_BI1, OUTPUT); pinMode(MF_BI2, OUTPUT);
  pinMode(MR_PWMA, OUTPUT); pinMode(MR_AI1, OUTPUT); pinMode(MR_AI2, OUTPUT);
  pinMode(MR_PWMB, OUTPUT); pinMode(MR_BI1, OUTPUT); pinMode(MR_BI2, OUTPUT);
  pinMode(LED_PIN, OUTPUT);

  // WiFi
  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("\nWiFi connected.");
  Serial.println(WiFi.localIP());
  server.begin();
}

void loop() {
  WiFiClient client = server.available();

  if (client) {
    String currentLine = "";
    while (client.connected()) {
      // 1. Safety Watchdog
      if (millis() - lastCmdTime > WATCHDOG_TIMEOUT) {
        stopMotors();
        digitalWrite(LED_PIN, LOW);
      }

      if (client.available()) {
        char c = client.read();
        currentLine += c;
        
        // 2. Parse Command: Expect format "V:x,y,w" (e.g., V:100,50,0) ends with newline
        if (c == '\n') {
          processCommand(currentLine);
          currentLine = "";
          lastCmdTime = millis();
          digitalWrite(LED_PIN, HIGH);
        }
      }
    }
    client.stop();
  } else {
    // No client connected? Stop motors.
    stopMotors();
    digitalWrite(LED_PIN, LOW);
  }
}

void processCommand(String cmd) {
  cmd.trim(); // Remove whitespace
  
  if (cmd.startsWith("V:")) {
    // Parse format V:vx,vy,omega
    int firstComma = cmd.indexOf(',');
    int secondComma = cmd.lastIndexOf(',');
    
    if (firstComma > 0 && secondComma > 0) {
      String sX = cmd.substring(2, firstComma);
      String sY = cmd.substring(firstComma + 1, secondComma);
      String sW = cmd.substring(secondComma + 1);
      
      int x = sX.toInt();
      int y = sY.toInt();
      int w = sW.toInt();
      
      driveMecanum(x, y, w);
    }
  }
}

void driveMecanum(int x, int y, int rot) {
  // Kinematics based on your previous "Square Dance" logic
  // FL = +y +x +rot
  // FR = +y -x -rot
  // RL = +y -x +rot
  // RR = +y +x -rot
  
  int fl = y + x + rot;
  int fr = y - x - rot;
  int rl = y - x + rot;
  int rr = y + x - rot;

  setMotor(MF_PWMB, MF_BI1, MF_BI2, fl); // FL
  setMotor(MF_PWMA, MF_AI1, MF_AI2, fr); // FR
  setMotor(MR_PWMB, MR_BI1, MR_BI2, rl); // RL
  setMotor(MR_PWMA, MR_AI1, MR_AI2, rr); // RR
}

void setMotor(int pwmPin, int in1, int in2, int speed) {
  // Clamp speed -255 to 255
  if (speed > 255) speed = 255;
  if (speed < -255) speed = -255;

  if (speed > 0) {
    digitalWrite(in1, HIGH);
    digitalWrite(in2, LOW);
    analogWrite(pwmPin, speed);
  } else if (speed < 0) {
    digitalWrite(in1, LOW);
    digitalWrite(in2, HIGH);
    analogWrite(pwmPin, -speed); // Make positive for PWM
  } else {
    digitalWrite(in1, LOW);
    digitalWrite(in2, LOW);
    analogWrite(pwmPin, 0);
  }
}

void stopMotors() {
  driveMecanum(0, 0, 0);
}
