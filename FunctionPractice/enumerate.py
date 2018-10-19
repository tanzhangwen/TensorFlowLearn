def HelloWorld():
	str = 'adggf'
	list3 = [1,2,4]
	tupl3 = (5, 8,2)
	set1 = {'ab','cb'}
	print(list(enumerate(str)))
	print(list(enumerate(str,start=2)))
	print(list(enumerate(list3,start=1)))
	print(list(enumerate(tupl3,start=1)))
	print(list(enumerate(set1,start=1)))

if __name__=="__main__":
    HelloWorld()
