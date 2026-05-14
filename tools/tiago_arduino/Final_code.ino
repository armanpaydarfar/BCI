/* Tiago Gunter 
 Last Updated 07 JAN 2026
*/

#include <elapsedMillis.h>
#include <Servo.h>
elapsedMillis timeElapsed; 

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
int RPWM = 5;   // Extension Pin
int LPWM = 6;   // Retraction Pin
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

  

  // CALIBRATING SERVO
  Serial.println("Initializing rotational motor...");
  calibrateServo();
  

  // CALIBRATING LINEAR ACTUATOR
  Serial.println("Initializing linear motor...");
  calibrateLinAct();


  // Calibration Complete
  Serial.println("Calibration complete. Waiting for next command. Enter: 'analog_value,angle,delay'.");
}




void loop() {
  while (Serial.available() > 0) {
    // Parsing input values, splitting them into analog (linear) and angle
    String input = Serial.readStringUntil('\n');
    input.trim();

    // Discrete HOME command: retract actuator to home and center servo.
    // Issued by the BCI driver at end of each successful MI trial; mirrors
    // Harmony's GO-then-later-HOME pattern in concept.
    if (input == "h") {
      Serial.println("HOME command received.");
      servo1.write(s_center);
      currentAngle = s_center;
      delay(500);
      moveToLimit(-1);
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


