all:
	$(MAKE) all -C efficientdet/src

clean:
	rm efficientdet/src/efficientdet_demo
	rm efficientdet/src/efficientdet_demo_gpu
