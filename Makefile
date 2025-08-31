###########################################################################
# Global targets
###########################################################################

quality-check:
#	@$(MAKE) -C algebraic_predictor quality-check
	@$(MAKE) -C mnist_predictor quality-check

run-test:
#	@$(MAKE) -C algebraic_predictor run-test
#	@$(MAKE) -C mnist_predictor run-test

format:
#	@$(MAKE) -C algebraic_predictor format
	@$(MAKE) -C mnist_predictor format

run:
#	@$(MAKE) -C algebraic_predictor run
	@$(MAKE) -C mnist_predictor run