all:
	$(MAKE) efficientdet -C efficientdet/src
	$(MAKE) efficientdet-gpu -C efficientdet/src
