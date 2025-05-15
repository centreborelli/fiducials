


def fiducial_cvqr(x, s):
	# pip install opencv-python
	import cv2
	return not not cv2.QrCodeDetector().detectAndDecode(x)[0]

def fiducial_pyzbar(x, s):
	# apt-get install libzbar0
	# pip install pyzbar
	import pyzbar.pyzbar
	return not not pyzbar.pyzbar.decode(x)


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
	x = iio.read(i)
	b = f(x, s)
	from sys import exit
	exit(not b)

version = 9
