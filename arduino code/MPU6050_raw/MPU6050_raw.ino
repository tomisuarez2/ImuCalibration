#include "I2Cdev.h"
#include "MPU6050.h"

/* Use "EXECUTE_READ" to read data from the MPU, otherwise it won't do it */
#define EXECUTE_READ

/* Defined here desired samplig rate frequency in Hz */
#define SMPL_RT_FREQ 100

/* MPU6050 default I2C address is 0x68*/
MPU6050 mpu;

/* OUTPUT FORMAT DEFINITION----------------------------------------------------------------------------------
- Use "OUTPUT_READABLE_ACCELGYRO" if you want to see a tab-separated list of the accel
X/Y/Z and gyro X/Y/Z values in decimal. Easy to read, but not so easy to parse, and slower over UART.

- Use "OUTPUT_BINARY_ACCELGYRO" to send all 6 axes of data as 16-bit binary, one right after the other.
As fast as possible without compression or data loss, easy to parse, but impossible to read for a human.
This output format is used as an output.
--------------------------------------------------------------------------------------------------------------*/

#define OUTPUT_READABLE_ACCELGYRO
//#define OUTPUT_BINARY_ACCELGYRO

int16_t ax, ay, az;
int16_t gx, gy, gz;
bool blinkState;
bool newData = false;
uint8_t intStatus;

void setup() {
  /*--Start I2C interface--*/
  #if I2CDEV_IMPLEMENTATION == I2CDEV_ARDUINO_WIRE
    Wire.begin();
  #elif I2CDEV_IMPLEMENTATION == I2CDEV_BUILTIN_FASTWIRE
    Fastwire::setup(400, true);
  #endif

  /* Initialize UART communication */
  Serial.begin(38400); //Initializate Serial wo work well at 8MHz/16MHz

  /* Initialize device and check connection */
  Serial.println("Initializing MPU...");
  mpu.initialize();
  Serial.println("Testing MPU6050 connection...");
  if(mpu.testConnection() ==  false){
    Serial.println("MPU6050 connection failed");
    while(true);
  }
  else{
    Serial.println("MPU6050 connection successful");
  }

  /* Set digital low pass filter mode */
  Serial.println("Setting DLPF mode...");
  mpu.setDLPFMode(0);

  /* Get digital low pass filter mode */
  Serial.println("Getting DLPF mode...");
  uint8_t mpuDLFPMode = mpu.getDLPFMode();
  Serial.println(mpuDLFPMode);

  #ifdef EXECUTE_READ
  /* Enabling interruptions */
  Serial.println("Enabling interruptions...");
  uint8_t intStatus = mpu.getIntStatus(); // Clear the status.
  mpu.setIntFreefallEnabled(0);
  mpu.setIntMotionEnabled(0);
  mpu.setIntZeroMotionEnabled(0);
  mpu.setIntFIFOBufferOverflowEnabled(0);
  mpu.setIntI2CMasterEnabled(0);
  mpu.setIntDataReadyEnabled(1);
  mpu.setInterruptMode(0);
  mpu.setInterruptDrive(0);
  mpu.setInterruptLatch(1);
  mpu.setInterruptLatchClear(0);
  uint8_t enabledInterrupt = mpu.getIntEnabled();
  Serial.print("The enabled interruptions on the MPU6050 are: ");
  Serial.println(enabledInterrupt);
  /* Enable interrupts in digital pin 2 of Arduino */
  Serial.println("Enabling Arduino interruptions on digital pin 2...");
  attachInterrupt(digitalPinToInterrupt(2), sendDataFunction, RISING);
  #endif

  /* Set sample frequency */
  Serial.println("Setting sampling frequency...");
  uint16_t gyroSampleRateFrequency;
  if (!mpuDLFPMode | mpuDLFPMode == 7) {
    /* Base sampling rate frequency in Hz */
    gyroSampleRateFrequency = 8000;
  }
  else{
    gyroSampleRateFrequency = 1000;
  }
  uint8_t mpuDesiredSampleRateDividerRegister = gyroSampleRateFrequency/SMPL_RT_FREQ - 1;
  mpu.setRate(mpuDesiredSampleRateDividerRegister);

  /* Get SMPLRT_DIV register */
  Serial.println("Getting SMPLRT_DIV register...");
  uint8_t mpuSampleRateDivRegister = mpu.getRate();
  Serial.println(mpuSampleRateDivRegister);
  Serial.println("Computing sample rate frequency...");
  Serial.print("Fs: ");
  Serial.println(gyroSampleRateFrequency/(1 + mpuSampleRateDivRegister));

  /* Waiting for confirmation */
  uint8_t proceed = 0;
  while(!proceed){
    if (Serial.available() > 0) {
      proceed = 1;
    }
  }

  /* Procceding to get raw data */
  Serial.println("Getting raw data...");

  /*Configure board LED pin for output*/
  pinMode(LED_BUILTIN, OUTPUT);
}
void loop() {
  if (newData){
    /* Read raw accel/gyro data from the module */
    mpu.getMotion6(&ax, &ay, &az, &gx, &gy, &gz);
    /* Print the obtained data on the defined format */
    #ifdef OUTPUT_READABLE_ACCELGYRO
      Serial.print(ax); Serial.print(",");
      Serial.print(ay); Serial.print(",");
      Serial.print(az); Serial.print(",");
      Serial.print(gx); Serial.print(",");
      Serial.print(gy); Serial.print(",");
      Serial.println(gz);
    #endif

    #ifdef OUTPUT_BINARY_ACCELGYRO
      Serial.write((uint8_t)(ax >> 8)); Serial.write((uint8_t)(ax & 0xFF));
      Serial.write((uint8_t)(ay >> 8)); Serial.write((uint8_t)(ay & 0xFF));
      Serial.write((uint8_t)(az >> 8)); Serial.write((uint8_t)(az & 0xFF));
      Serial.write((uint8_t)(gx >> 8)); Serial.write((uint8_t)(gx & 0xFF));
      Serial.write((uint8_t)(gy >> 8)); Serial.write((uint8_t)(gy & 0xFF));
      Serial.write((uint8_t)(gz >> 8)); Serial.write((uint8_t)(gz & 0xFF));
    #endif
    /* Clear the interrupt status */
    intStatus = mpu.getIntStatus();
    newData = false;
    /* Blink LED to indicate activity */
    blinkState = !blinkState;
    digitalWrite(LED_BUILTIN, blinkState);
  }
}

#ifdef EXECUTE_READ
void sendDataFunction() {
  newData = true;
}
#endif
