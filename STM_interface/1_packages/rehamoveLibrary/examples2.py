from rehamove import *          # Import our library
import time
import math 

# r.change_mode(0)                # Change to low-level mode (each pulse is sent separtely)

FES_port = "/dev/ttyUSB0"
r = Rehamove(FES_port)    # Open USB port (on Windows)
I_STS_proxExt = 5	# in mA for sensory-threshold stimulation at the proximal location for the extensor muuscle
I_STS_distExt = 3.5	# in mA for sensory-threshold stimulation at the distal location for the extensor muuscle
I_MTS_proxExt = 8.5	# in mA for motor-threshold stimulation at the proximal location for the extensor muuscle
I_MTS_distExt = 9	# in mA for motor-threshold stimulation at the distal location for the extensor muuscle

I_STS_proxFlex = 5	# in mA for sensory-threshold stimulation at the proximal location for the flexor muuscle
I_STS_distFlex = 4.5 	# in mA for sensory-threshold stimulation at the distal location for the flexor muuscle
I_MTS_proxFlex = 7		 	# in mA for motor-threshold stimulation at the proximal location for the flexor muuscle
I_MTS_distFlex = 7	 	# in mA for motor-threshold stimulation at the distal location for the flexor muuscle


FES_freq = 30 	# in Hz
FES_pulseWidth = 250
distalChannel_flex = 'white'
proximalChannel_flex = 'black'
distalChannel_ext = 'red'
proximalChannel_ext = 'blue'


STS_duration = [3,6,9] 	# in seconds for distal, both , proximal

t_init = time.time()
t_passed = time.time()
# stimulation sfreq: low sensory and high muscular; while stimulating, you can move the elecrrodes to try to elicit better movement: might need to adjust the proximality and distality of 
# never go above 30mA (the first intensity that is felt: larger than if not sure) at least 1 level belo muscluar contraction
ext = 1


if ext==2:

	while t_passed - t_init < STS_duration[0]:
		t_passed = time.time()
		r.pulse(distalChannel_ext,I_STS_distExt5, FES_pulseWidth) # 3 250+int(((t_passed - t_init)/STS_duration[0])*250)
		r.pulse(proximalChannel_ext,I_STS_proxExt, FES_pulseWidth) # 3 250+int(((t_passed - t_init)/STS_duration[0])*250)
		r.pulse(proximalChannel_flex,I_STS_proxFlex, FES_pulseWidth) # 3 250+int(((t_passed - t_init)/STS_duration[0])*250)
		r.pulse(proximalChannel_flex,I_STS_distFlex, FES_pulseWidth) # 3 250+int(((t_passed - t_init)/STS_duration[0])*250)
		# r.pulse(distalChannel_ext,3.5, 250+int(((t_passed - t_init)/STS_duration[0])*250)) # 3 150+int(((t_passed - t_init)/STS_duration[1])*250)
		time.sleep(1/30)	


if ext==1:
	while t_passed - t_init < STS_duration[0]:
		t_passed = time.time()
		# r.pulse(distalChannel_ext,I_STS_distExt, FES_pulseWidth) # 3 250+int(((t_passed - t_init)/STS_duration[0])*250)
		r.pulse(distalChannel_ext,I_STS_proxExt, FES_pulseWidth) # 3 250+int(((t_passed - t_init)/STS_duration[0])*250)
		# r.pulse(distalChannel_ext,3.5, 250+int(((t_passed - t_init)/STS_duration[0])*250)) # 3 150+int(((t_passed - t_init)/STS_duration[1])*250)
		time.sleep(1/30)
	while t_passed - t_init < STS_duration[1]:
	# 	t_passed = time.time()
	# 	r.pulse(distalChannel_ext,I_STS_distExt, FES_pulseWidth) # 3 250+int(((t_passed - t_init)/STS_duration[0])*250)
	# 	r.pulse(proximalChannel_ext,I_STS_proxExt, FES_pulseWidth) # 3 250+int(((t_passed - t_init)/STS_duration[0])*250)
	# 	# r.pulse(distalChannel_ext,3.5, 250+int(((t_passed - t_init)/STS_duration[0])*250)) # 3 150+int(((t_passed - t_init)/STS_duration[1])*250)
	# 	time.sleep(1/30)	
	# while t_passed - t_init < STS_duration[2]:
	# 	t_passed = time.time()
	# 	r.pulse(proximalChannel_ext,I_MTS_proxExt, FES_pulseWidth) # 3 250+int(((t_passed - t_init)/STS_duration[0])*250)
	# 	r.pulse(proximalChannel_ext,I_MTS_distExt, FES_pulseWidth) # 3 250+int(((t_passed - t_init)/STS_duration[0])*250)
	# 	# r.pulse(distalChannel_ext,3.5, 250+int(((t_passed - t_init)/STS_duration[0])*250)) # 3 150+int(((t_passed - t_init)/STS_duration[1])*250)
	# 	time.sleep(1/30)	

	# while t_passed - t_init < STS_duration[1]:
	# 	t_passed = time.time()
	# 	# r.pulse(distalChannel_ext,6, 450) # 3 150+int(((t_passed - t_init)/STS_duration[1])*250)
	# 	r.pulse(proximalChannel_ext, 5, 500) # 3
	# 	time.sleep(1/30)	
	# while t_passed - t_init < STS_duration[0]:
	# 	t_passed = time.time()
	# 	r.pulse(proximalChannel_ext,4.5, 350) # 3
	# 	time.sleep(1/40)	

	# while t_passed - t_init < STS_duration[1]:
	# 	t_passed = time.time()
	# 	# r.pulse(distalChannel_ext, 3 ,500) # 3
	# 	r.pulse(proximalChannel_ext, 4 , 500) # 4.5
	# 	time.sleep(1/40)

	# while t_passed - t_init < STS_duration[2]:
	# 	t_passed = time.time()
	# 	# r.pulse(distalChannel_ext, 6.5 ,500) # 6.5
	# 	r.pulse(proximalChannel_ext, 5 , 500)  # 6
	# 	time.sleep(1/35)
else:
	while t_passed - t_init < STS_duration[0]:
		t_passed = time.time()
		r.pulse(distalChannel_flex,I_STS_distFlex, FES_pulseWidth) # 3 250+int(((t_passed - t_init)/STS_duration[0])*250)
		# r.pulse(distalChannel_ext,3.5, 250+int(((t_passed - t_init)/STS_duration[0])*250)) # 3 150+int(((t_passed - t_init)/STS_duration[1])*250)
		time.sleep(1/30)
	while t_passed - t_init < STS_duration[1]:
		t_passed = time.time()
		r.pulse(distalChannel_flex,I_STS_distFlex, FES_pulseWidth) # 3 250+int(((t_passed - t_init)/STS_duration[0])*250)
		r.pulse(proximalChannel_flex,I_STS_proxFlex, FES_pulseWidth) # 3 250+int(((t_passed - t_init)/STS_duration[0])*250)
		# r.pulse(distalChannel_ext,3.5, 250+int(((t_passed - t_init)/STS_duration[0])*250)) # 3 150+int(((t_passed - t_init)/STS_duration[1])*250)
		time.sleep(1/30)	
	while t_passed - t_init < STS_duration[2]:
		t_passed = time.time()
		r.pulse(distalChannel_flex,I_MTS_distFlex, FES_pulseWidth) # 3 250+int(((t_passed - t_init)/STS_duration[0])*250)
		r.pulse(proximalChannel_flex,I_MTS_proxFlex, FES_pulseWidth) # 3 250+int(((t_passed - t_init)/STS_duration[0])*250)
		# r.pulse(distalChannel_ext,3.5, 250+int(((t_passed - t_init)/STS_duration[0])*250)) # 3 150+int(((t_passed - t_init)/STS_duration[1])*250)
		time.sleep(1/30)	

	# while t_passed - t_init < STS_duration[0]:
	# 	t_passed = time.time()
	# 	r.pulse(distalChannel_flex, 3.5 , 500) # 3.5
	# 	time.sleep(1/30)	

	# while t_passed - t_init < STS_duration[1]:
	# 	t_passed = time.time()
	# 	r.pulse(proximalChannel_flex, 3 ,500) # 3
	# 	r.pulse(distalChannel_flex, 3.5 , 500) # 3.5
	# 	time.sleep(1/30)


	# while t_passed - t_init < STS_duration[2]:
	# 	t_passed = time.time()
	# 	r.pulse(proximalChannel_flex, 4 ,500) # 4
	# 	r.pulse(distalChannel_flex, 4.5 , 520) # 4.5

	# 	time.sleep(1/30)

