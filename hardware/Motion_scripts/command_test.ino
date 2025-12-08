#include <WiFi.h>
#include "soc/soc.h"
#include "soc/rtc_cntl_reg.h"

// --- CONFIGURATION ---
const char* ssid = "Thousand-sunny";
const char* password = "ZiaBHai123";
int speed_val = 150; // Speed (0-255)
// ---------------------

// --- PIN DEFINITIONS (Mecanum) ---
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

WiFiServer server(80);

// State Variables
bool isMoving = false;
unsigned long moveEndTime = 0;
const int moveDuration = 1000; // 1 second duration

// Function Prototypes
void moveForward();
void moveBackward();
void moveRight();
void moveLeft();
void stopMotors();
void setMotor(int pwmPin, int in1, int in2, int speed, bool forward);

void setup() {
  WRITE_PERI_REG(RTC_CNTL_BROWN_OUT_REG, 0); // Brownout fix
  
  // Motor Pins Setup
  pinMode(MF_PWMA, OUTPUT); pinMode(MF_AI1, OUTPUT); pinMode(MF_AI2, OUTPUT);
  pinMode(MF_PWMB, OUTPUT); pinMode(MF_BI1, OUTPUT); pinMode(MF_BI2, OUTPUT);
  pinMode(MR_PWMA, OUTPUT); pinMode(MR_AI1, OUTPUT); pinMode(MR_AI2, OUTPUT);
  pinMode(MR_PWMB, OUTPUT); pinMode(MR_BI1, OUTPUT); pinMode(MR_BI2, OUTPUT);
  
  pinMode(LED_PIN, OUTPUT);
  Serial.begin(115200);

  // WiFi Setup
  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("\nWiFi connected.");
  Serial.println(WiFi.localIP());
  digitalWrite(LED_PIN, HIGH);
  server.begin();
}

void loop() {
  // 1. CHECK FOR COMMANDS
  WiFiClient client = server.available();
  if (client) {
    while (client.connected()) {
      if (client.available()) {
        char c = client.read();
        
        // Direct Character Checks for Instant Response
        if (c == '1') {
           Serial.println("CMD: RIGHT (1s)");
           moveRight();
           isMoving = true;
           moveEndTime = millis() + moveDuration;
        }
        else if (c == '2') {
           Serial.println("CMD: FORWARD (1s)");
           moveForward();
           isMoving = true;
           moveEndTime = millis() + moveDuration;
        }
        else if (c == '3') {
           Serial.println("CMD: LEFT (1s)");
           moveLeft();
           isMoving = true;
           moveEndTime = millis() + moveDuration;
        }
        else if (c == '4') {
           Serial.println("CMD: BACKWARD (1s)");
           moveBackward();
           isMoving = true;
           moveEndTime = millis() + moveDuration;
        }
      }
    }
    client.stop();
  }

  // 2. HANDLE AUTO-STOP (Non-blocking)
  if (isMoving) {
    if (millis() > moveEndTime) {
      stopMotors();
      isMoving = false;
      Serial.println("Auto-Stop triggered");
    }
  }
}

// --- MOTOR HELPER FUNCTIONS ---

void setMotor(int pwmPin, int in1, int in2, int speed, bool forward) {
  analogWrite(pwmPin, speed);
  if (forward) {
    digitalWrite(in1, HIGH);
    digitalWrite(in2, LOW);
  } else {
    digitalWrite(in1, LOW);
    digitalWrite(in2, HIGH);
  }
}

void stopMotors() {
  analogWrite(MF_PWMA, 0); analogWrite(MF_PWMB, 0);
  analogWrite(MR_PWMA, 0); analogWrite(MR_PWMB, 0);
}

// FL=FrontLeft, FR=FrontRight, RL=RearLeft, RR=RearRight
void moveForward() {
  setMotor(MF_PWMB, MF_BI1, MF_BI2, speed_val, true);  // FL Fwd
  setMotor(MF_PWMA, MF_AI1, MF_AI2, speed_val, true);  // FR Fwd
  setMotor(MR_PWMB, MR_BI1, MR_BI2, speed_val, true);  // RL Fwd
  setMotor(MR_PWMA, MR_AI1, MR_AI2, speed_val, true);  // RR Fwd
}

void moveBackward() {
  setMotor(MF_PWMB, MF_BI1, MF_BI2, speed_val, false); // FL Rev
  setMotor(MF_PWMA, MF_AI1, MF_AI2, speed_val, false); // FR Rev
  setMotor(MR_PWMB, MR_BI1, MR_BI2, speed_val, false); // RL Rev
  setMotor(MR_PWMA, MR_AI1, MR_AI2, speed_val, false); // RR Rev
}

void moveRight() {
  setMotor(MF_PWMB, MF_BI1, MF_BI2, speed_val, true);  // FL Fwd
  setMotor(MF_PWMA, MF_AI1, MF_AI2, speed_val, false); // FR Rev
  setMotor(MR_PWMB, MR_BI1, MR_BI2, speed_val, false); // RL Rev
  setMotor(MR_PWMA, MR_AI1, MR_AI2, speed_val, true);  // RR Fwd
}

void moveLeft() {
  setMotor(MF_PWMB, MF_BI1, MF_BI2, speed_val, false); // FL Rev
  setMotor(MF_PWMA, MF_AI1, MF_AI2, speed_val, true);  // FR Fwd
  setMotor(MR_PWMB, MR_BI1, MR_BI2, speed_val, true);  // RL Fwd
  setMotor(MR_PWMA, MR_AI1, MR_AI2, speed_val, false); // RR Rev
}
