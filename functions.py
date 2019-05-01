"coding utf-8"


import cv2 as cv 
import numpy as np 
import face_recognition as fr 
#from PIL import Image, ImageDraw, ImageFilter
import operator as op
from operator import itemgetter
import pdb	# for debugging




def reverse_rgb(x):

	""" This function returns a rgb value as an array from a 
		digit between 0 and 16777215, to do so the digit is decomposed  as a 
		255 base number.

	Variables index :
		x : The digit we gotta convert
		rgb : The returned array containing a rgb value
		v : The values used to fill the array
	"""

	rgb = []
	i = 0
	while len(rgb) != 3:
	
		if len(rgb) == 2:
			rgb.append(x)

		else:
			k = 0

			while k*256**(2-i) < x:

				k += 1

			if k == 0 :

				v = k

			else:

				v = k - 1

			rgb.append(v)
			x -= v*256**(2-i)
			i += 1

	return rgb


def dominant_colour(face, landmark):

	"""
		This functions returns the color skin of a face, 
		which is the most common colour. To do so, we convert 
		the face's array by turning the (r, g, b) pixels
		values into (r*256**2 + g*256**1 + b*256**0) to get a 
		single channel array that still contains all the infos
		of the image, once we found the most common value, we go
		backwards by putting it into its (r, g, b) form.

		variables index:
			face: The face wa gonna erase
			roi: region of interest within the face to get the color skin, it's basically the nose,
			roi_1c : our roi converted into a 1 channel table (1c = 1 channel)
			mc : most common value found in face_1c, will be between 0 and 16777215
			skin : the (r, g, b) skin colour, which is mc reversed


	"""

	# First of all we gotta define our ROI thanks to the face landmark

	left, top = len(face), len(face[0])
	right, bottom = 0, 0

	for pt in landmark:

		if pt[0] < left:
			left = pt[0]

		if pt[0] > right:
			right = pt[0]

		if pt[1] < top:
			top = pt[1]

		if pt[1] > bottom:
			bottom = pt[1]

	# Sometimes we have left = right which can lead to a crash so let's just do that to avoid it:

	if left == right:

		left -= 2
		right +=2

	

	roi = face[top:bottom, left:right]


	# now we convert the roi array

	roi_1c = []
	

	for i, vi in enumerate(roi):
		for j, vj in enumerate(vi):

			roi_1c.append(256**2*vj[0] + 256*vj[1] + vj[2])

	# Now let's find the most mommon value and reverse it

	mc =  np.bincount(roi_1c).argmax()

	skin = reverse_rgb(mc)
	#print(skin)
	skin = np.array([skin[0], skin[1], skin[2]])

	return (np.asscalar(skin[0]), np.asscalar(skin[1]), np.asscalar(skin[2]))


def roi_shift(landmarks, x, y):

	"""
		This function is required to put back the landmarks
		at the right place because of the region of interests.
		variable index :
			landmarks : The landmarks of the current face
			x, y : The coordinates of our ROI in the original frame
			c_landmarks : The corrected landmarks


	"""

	c_landmarks = {}

	for k, tab in landmarks.items():
		t = []
		for c in tab:
			t.append((c[0] + x, c[1] + y))  
		c_landmarks[k] = t

	return c_landmarks


def mouth(top, bot):

	"""
		This function is required to make landmarks for the whole mouth, not just
		the lips.

		Variables index:

	"""
	t_left = min(top, key = itemgetter(0))
	b_left = min(bot, key = itemgetter(0))

	t_right = max(top, key = itemgetter(0))
	b_right = max(bot, key = itemgetter(0))

	m = []
	m.append(t_left)
	m.append(t_right)
	#m.append(b_left)
	#m.append(b_right)

	for pt in top:

		if pt[1] > t_left[1] or pt[1] > t_right[1]:

			m.append(pt)

	for pt in bot:

		if pt[1] < b_left[1] or pt[1] < b_right[1]:

			m.append(pt)

	print("{}\n\n".format(m))

	return m


def set_roi(landmarks):

	"""
		This function is here to make rois in order to 
		filter edges to make the final result smoother
		No variables index, it's trivial here.

	"""

	left = min(landmarks, key = itemgetter(0))[0]
	right = max(landmarks, key = itemgetter(0))[0]
	top = min(landmarks, key = itemgetter(1))[1]
	bottom = max(landmarks, key = itemgetter(1))[1]



	return left, right, top, bottom


def homothetia(polygon, nr):

	""" As the title says, this function is to perform an 
		homothetia, it is required because some of the landmarks are
		a bit too small

	"""
	

	# First we need the Barycentre of the polygon (the landmarks)
	#pdb.set_trace()
	bary = np.mean(polygon, axis = 0)
	bary = np.array(bary, dtype = np.uint16)
	


	# Now we can make our homothetia 

	new_poly = []	# This will be our new polygon
	
	for pt in polygon:

		vc = tuple(map(op.sub, pt, bary))		# This is the vector that goes from the barycentre to the current point
		vc = tuple(int(k*nr) for k in vc)		# We change the norm of this vector thanks to nr
		p = tuple(map(op.add, vc, bary))		# And add its coordinates with the barycentre to get the new point
		new_poly.append(p)
		

	new_poly = np.array(new_poly)


	return np.array(new_poly)




def erase_face(faces):

	"""This is where you gonna loose your face. The idea is to find
		The mouth, the nose, the eyes, etc and replace all the pixel values 
		by the skin colour value.

		Variables index :
			faces : The current frame
			faces_list : the detected faces in the current frame
			colour : he skin colour of the current face
			pil_face : the face under a PIL object form
			d : a PIL object we need to draw on the face

	"""

	# First let's find the faces thanks to face_detection

	faces_list = fr.face_locations(faces)

	# We use the PIL library to "erase" faces, to do so, we gotta convert the image into a PIL object and create a PIL draing object
	pil_face = Image.fromarray(faces)
	d = ImageDraw.Draw(pil_face, 'RGBA')

	for f in faces_list:

		# the region of interest on the frame
		top, right, bottom, left = f
		roi = faces[top:bottom, left:right]	

		# the face features of the person

		f_landmarks = fr.face_landmarks(roi)	

		if len(f_landmarks) > 0:		# This is to prevent any crash in case no one is detected

	    	# The skin color of the person
			colour = dominant_colour(roi, f_landmarks[0]['nose_bridge'])
			#print(colour)

			f_landmarks = roi_shift(f_landmarks[0], left, top)

			#lips = list(set(f_landmarks['top_lip']) | set(f_landmarks['bottom_lip']))
			#lips = mouth(f_landmarks['top_lip'], f_landmarks['bottom_lip'])

			#now we can hide him, or her

			d.polygon(f_landmarks['left_eye'], fill=colour)
			d.polygon(f_landmarks['left_eyebrow'], fill=colour)
			d.polygon(f_landmarks['right_eye'], fill=colour)
			d.polygon(f_landmarks['right_eyebrow'], fill=colour)
			d.polygon(f_landmarks['top_lip'], fill=colour)
			d.polygon(f_landmarks['bottom_lip'], fill=colour)
			d.polygon(f_landmarks['nose_tip'], fill=colour)
			d.polygon(f_landmarks['nose_bridge'], fill=colour)
			#d.polygon(f_landmarks['chin'], fill=colour)
			#d.polygon(lips, fill=colour)


			# Filtering 

			# First we gotta define our filter, it's a basic mean filter
			#kernel = np.ones((15, 15),np.float32)/225

			ln, rn, tn, bn = set_roi(f_landmarks['nose_tip'] + f_landmarks['nose_bridge'])
			pil_face[tn:bn, ln, rn].filter(ImageFilter.BLUR)

			


		


	return np.array(pil_face)


def erase_face2(faces):

	"""This is where you gonna loose your face. The idea is to find
		The mouth, the nose, the eyes, etc and replace all the pixel values 
		by the skin colour value.

		Variables index :
			faces : The current frame
			faces_list : the detected faces in the current frame
			clr : he skin colour of the current face
			pil_face : the face under a PIL object form
			d : a PIL object we need to draw on the face

	"""

	# First let's find the faces thanks to face_detection

	faces_list = fr.face_locations(faces)
			
	# Filtering 
	# First we gotta define our filter, it's a basic mean filter
	n = 5
	kernel = np.ones((n, n),np.float32)/n**2


	for f in faces_list:

		# the region of interest on the frame
		top, right, bottom, left = f
		roi = faces[top:bottom, left:right]	

		# the face features of the person

		f_landmarks = fr.face_landmarks(roi)	

		if len(f_landmarks) > 0:		# This is to prevent any crash in case no one is detected

	    	# The skin color of the person
			clr = dominant_colour(roi, f_landmarks[0]['nose_bridge'])
			
			
			f_landmarks = roi_shift(f_landmarks[0], left, top)

			#lips = list(set(f_landmarks['top_lip']) | set(f_landmarks['bottom_lip']))
			#lips = mouth(f_landmarks['top_lip'], f_landmarks['bottom_lip'])
			
			#now we can hide him, or her
			
			r_eye = homothetia(f_landmarks['right_eye'], 2)
			l_eye = homothetia(f_landmarks['left_eye'], 2)
			
			#cv.fillPoly(faces, pts = [np.array(f_landmarks['right_eye'], np.int32).reshape(-1, 1, 2)], color = clr)
			cv.fillPoly(faces, pts = [l_eye], color = clr)
			cv.fillPoly(faces, pts = [r_eye], color = clr)




			#ln, rn, tn, bn = set_roi(f_landmarks['nose_tip'] + f_landmarks['nose_bridge'])
			ln, rn, tn, bn = set_roi(r_eye)
			faces[tn:bn, ln:rn] = cv.filter2D(faces[tn:bn, ln:rn], -1, kernel)
			ln, rn, tn, bn = set_roi(l_eye)
			faces[tn:bn, ln:rn] = cv.filter2D(faces[tn:bn, ln:rn], -1, kernel)



	return faces





