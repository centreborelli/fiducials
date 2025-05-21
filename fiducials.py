
def eprint(*args, **kwargs):
	import sys
	print(*args, file=sys.stderr, **kwargs)


def fiducial_cvqr(x):
	# pip install opencv-python
	import cv2
	from numpy import uint8
	X = x.astype(uint8)
	#import iio
	#iio.write("/tmp/cvqr_x.npy", x)
	#iio.write("/tmp/cvqr_X.npy", X)
	r = cv2.QRCodeDetector().detectAndDecode(X)[0]
	eprint(f"x.shape={x.shape}, r={r}")
	return not not r

def fiducial_pyzbar(x):
	# apt-get install libzbar0
	# pip install pyzbar
	import pyzbar.pyzbar
	r = pyzbar.pyzbar.decode(x)
	eprint(f"x.shape={x.shape}, r={r}")
	return not not r


# visible API
fiducials = [ "cvqr", "pyzbar" ]


# print the install isntructions for all the dependencies
def printinstall():
	L = []
	with open(__file__) as f:
		L = [l.strip() for l in f]
	p = False
	for l in L:
		if l.startswith("def fiducial_"):
			p = True
		elif p and l.startswith("# "):
			print(f"RUN {l[2:]}")
		else:
			p = False


# unified interface for all the algorithms above
def G(m, x):
	""" check whether image x has a fiducial marking according to m """
	f = globals()[f"fiducial_{m}"]
	return f(x, σ)



# cli interfaces to the above functions
if __name__ == "__main__":
	from sys import argv as v
	def pick_option(o, d):
		if int == type(o): return v[o]
		return type(d)(v[v.index(o)+1]) if o in v else d
	if len(v) < 2 or v[1] not in fiducials:
		print(f"usage:\n\tfiducials {{{'|'.join(fiducials)}}}")
		exit(0)
	import iio
	i = pick_option("-i", "-")
	f = globals()[f"fiducial_{v[1]}"]
	x = iio.read(i).clip(0,255)
	b = f(x)
	from sys import exit
	exit(not b)

version = 9
