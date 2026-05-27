/* Tiago Gunter 
 Last Updated 07 JAN 2026
*/

#include <Servo.h>

/////////////////     SERVO     /////////////////
// Initiating Servo
Servo servo1;

// Servo Range
int s_center = 39; // Center of Servo range
int s_r_max = 9; // Max right of Servo range
int s_l_max = 69; // Max left of Servo range

int delayTime = 150;
int pos;
int currentAngle;
/////////////////////////////////////////////////




/////////////////     LIN ACT     /////////////////
// Initiating Lin Actuator
int RPWM = 6;   // Extension Pin
int LPWM = 5;   // Retraction Pin
int sensorPin = A0;  // Feedback Pin

// Linear Actuator Range
int maxAnalogReading = 900;  //max limit with testing
int minAnalogReading = 350; //min limit with testing
// Max - Min = 7.4in extension

// Linear Actuator Speed
int Speed = 230; //  max speed: 255


int sensorVal; // Variable for Potentiometer Feedback value
///////////////////////////////////////////////////


int targetAnalog, targetAngle;
int comma1, comma2;

void setup() {

  // Servo Motor pin
  servo1.attach(2);

  // Linear Actuator pins
  pinMode(RPWM, OUTPUT); // Extend
  pinMode(LPWM, OUTPUT); // Retract
  pinMode(sensorPin, INPUT); // Feedback


  // Initializing
  Serial.begin(9600);
  Serial.println(" ");
  Serial.println("Begin");

  // Boot is now handshake-only: print a banner and wait for commands.
  // The Python side (Utils/tiagobot.open_port) treats this banner as
  // proof the device is alive. The 15-30 s actuator calibration sweep
  // moved into runCalibration(), triggered explicitly by the 't'
  // command from the operator (panel "Test" button) or driver.
  //
  // Rationale: prior behaviour auto-calibrated on every USB enumerate
  // and on every panel/driver port-open. The visible sweep slowed down
  // every session start. With handshake-only boot, the actuator stays
  // idle at whatever position it was last in until the operator
  // explicitly calibrates. Motion commands (letters, 'h') still work
  // pre-calibration using the hardcoded s_center / s_l/r_max defaults
  // — accuracy may be lower than post-calibration but the actuator
  // won't refuse to move.
  Serial.println("Tiagobot ready. Send '?' to ping, 't' to calibrate, "
                 "'analog,angle,delay' to GO, 'h' to HOME.");
}




void loop() {
  while (Serial.available() > 0) {
    // Parsing input values, splitting them into analog (linear) and angle
    String input = Serial.readStringUntil('\n');
    input.trim();

    // Handshake / ping. Python's open_port sends this immediately
    // after opening the serial port to confirm the device is alive
    // without triggering the slow actuator calibration sweep. Reply
    // is intentionally short so the round-trip is sub-second.
    if (input == "?") {
      Serial.println("OK");
      continue;
    }

    // Explicit calibration command. Operator triggers this via the
    // panel's Test button when they want to (re)home the actuator
    // and verify the full motion range. Same banner text the prior
    // setup()-driven calibration printed, so the Python wait logic
    // in Utils/tiagobot.calibrate() can latch onto the existing
    // CALIBRATION_READY_MARKER.
    if (input == "t") {
      Serial.println("Initializing rotational motor...");
      calibrateServo();
      Serial.println("Initializing linear motor...");
      calibrateLinAct();
      Serial.println("Calibration complete. Waiting for next command. "
                     "Enter: 'analog_value,angle,delay'.");
      continue;
    }

    // Discrete HOME command: retract actuator + center servo, with the
    // two motions running simultaneously, mirroring the GO loop's
    // driveServo+driveActuator pattern. Each iteration steps the servo
    // one degree toward s_center (paced by `delayTime`, set by the last
    // GO command) AND drives the retract PWM. Loop exits when the
    // linear actuator reaches its retracted home (sensorVal <=
    // minAnalogReading). If the servo hasn't reached s_center by then,
    // the next GO command's `servo1.write(s_center)` reset (line 123)
    // takes over — same behavior as the original auto-retract before
    // we split HOME out as a discrete command.
    if (input == "h") {
      Serial.println("HOME command received.");
      Serial.println("Retracting towards home . . .");
      sensorVal = analogRead(sensorPin);
      while (sensorVal > minAnalogReading) {
        driveServo(s_center, currentAngle);
        driveActuator(-1, Speed);
        displayOutput();
        sensorVal = analogRead(sensorPin);
      }
      driveActuator(0, Speed);
      Serial.println("Homed. Waiting for next command.");
      continue;
    }

    int comma1 = input.indexOf(',');
    int comma2 = input.indexOf(',', comma1 + 1);

    
    if (comma1 != -1 && comma2 != -1) {
      targetAnalog = input.substring(0, comma1).toInt();
      targetAngle = input.substring(comma1 + 1, comma2).toInt();
      delayTime = input.substring(comma2 + 1).toInt();


      //////////////////////////////////////////////////////////////////////////////////////
      // TESTING
      servo1.write(s_center);       // This makes the "currentAngle" become the center
      currentAngle = servo1.read(); // center
      Serial.print("The 'currentAngle' is: ");
      Serial.println(currentAngle);
      //////////////////////////////////////////////////////////////////////////////////////

      Serial.print("Target Analog Position: ");
      Serial.println(targetAnalog);
      Serial.print("Target Angle: ");
      Serial.println(targetAngle);
      Serial.print("Delay Time: ");
      Serial.println(delayTime);
    }

 
    if (targetAnalog >= minAnalogReading && targetAnalog <= maxAnalogReading) {
      //Serial.print("Moving to raw target value: ");
      //Serial.println(targetAnalog);

      sensorVal = analogRead(sensorPin);


      Serial.println("Moving towards target location . . .");
        
      while (sensorVal < targetAnalog) {
        driveServo(targetAngle,currentAngle);
        driveActuator(1, Speed);
        displayOutput();
        sensorVal = analogRead(sensorPin);
        
        }
      
      driveActuator(0, Speed);  // Stop actuator



      Serial.println("Target Location Reached.");
      Serial.println("Waiting for next command. Enter: 'analog_value,angle,delay' or 'h' to home.");


      // In case of invalid input
    } else {
      Serial.println("Invalid input. Enter raw analog value (350–750).");
      Serial.read(); // Clear junk
    }
  }

  
}




void moveToLimit(int Direction) {
  const int settleThreshold = 15;  // Acceptable range buffer
  int currReading;

  if (Direction == 1) {
    // EXTEND: Move until value <= low target (e.g., 10)
    driveActuator(1, Speed);
    while (true) {
      currReading = analogRead(sensorPin);
      Serial.print("Extending... Reading: ");
      Serial.println(currReading);

      if (maxAnalogReading - currReading  <=  settleThreshold) {
        Serial.println("Reached full extension.");
        break;
      }

    }
  } else if (Direction == -1) {
    // RETRACT: Move until value >= high target (e.g., 1020)
    driveActuator(-1, Speed);
    while (true) {
      currReading = analogRead(sensorPin);
      Serial.print("Retracting... Reading: ");
      Serial.println(currReading);

      if (currReading - minAnalogReading <=  settleThreshold) {
        Serial.println("Reached full retraction.");
        break;
      }
      // int maxAnalogReading = 870;  //max limit with testing
      // int minAnalogReading = 400;

    }
  } else {
    Serial.println("Invalid direction! Must be -1 (retract) or 1 (extend).");
    return;
  }

  driveActuator(0, Speed); // Stop
}



void homeActuators() {
  delay(2000);
  servo1.write(s_center);
  delay(500);
  moveToLimit(-1);
  
  
}


// Function to display Linear Actuator and Servo motor positions
void displayOutput(){
  sensorVal = analogRead(sensorPin);
  currentAngle = servo1.read();
    Serial.print("Linear Analog Reading: ");
    Serial.print(sensorVal);
    Serial.print(" || Angle Reading: ");
    Serial.println(currentAngle);

}

// Function to drve Linear Actuator
void driveActuator(int Direction, int Speed){ 
  switch(Direction){
    case 1:       //extension
      analogWrite(RPWM, Speed);
      analogWrite(LPWM, 0);
      break;
   
    case 0:       //stopping
      analogWrite(RPWM, 0);
      analogWrite(LPWM, 0);
      break;

    case -1:      //retraction
      analogWrite(RPWM, 0);
      analogWrite(LPWM, Speed);
      break;
  }
}

/*
// THIS WORKS, BUT IS NOT SIMULTANEOUS 13 DEC
void driveServo(int targetAngle){
  //servo1.write(s_center);
  currentAngle = servo1.read();
  if (currentAngle == targetAngle) return;
  delay(1000);

  if (targetAngle > currentAngle){
    for(pos = currentAngle; pos <= targetAngle ; pos += 1)
    {                                  
      servo1.write(pos);
      delay(120); // Slowing down the servo rotation             
    } 
  }

  else{
    for(pos = currentAngle; pos >= targetAngle ; pos -= 1)
    {                                  
      servo1.write(pos);
      delay(120); // Slowing down the servo rotation             
    } 
  }
}
*/


// TRYING SIMULTANEOUS MOTION 15 DEC
void driveServo(int targetAngle, int currentAngle){

  if (currentAngle == targetAngle) return;
  delay(100);

  if (targetAngle > currentAngle){
    currentAngle = currentAngle + 1;      
    servo1.write(currentAngle);
    displayOutput(); 
    delay(delayTime); // Slowing down the servo rotation             
   
  }

  else if (targetAngle < currentAngle){
    currentAngle = currentAngle - 1;
    servo1.write(currentAngle);
    displayOutput();           
    delay(delayTime); // Slowing down the servo rotation   
    
  }
}

// Function to Calibrate Servo motor
void calibrateServo(){
  servo1.write(s_r_max);
  delay(1000);
  for(pos = s_r_max; pos <= s_l_max ; pos += 1){            
    servo1.write(pos);
    displayOutput();    
    delay(100); // Slowing down the servo rotation             
  } 

  for(pos = s_l_max; pos >= s_center ; pos -= 1){                   
    servo1.write(pos);
    displayOutput();   
    delay(100); // Slowing down the servo rotation             
  } 
  

  servo1.write(s_center);
}

// Function to Linear Actuator
void calibrateLinAct(){

  moveToLimit(1);    // Extend fully : 750
  moveToLimit(-1);   // Retract fully : 350
}
