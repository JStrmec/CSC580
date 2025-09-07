###########################################################################
# Global targets
###########################################################################

quality-check:
#	@$(MAKE) -C algebraic_predictor quality-check
#	@$(MAKE) -C mnist_predictor quality-check
#	@$(MAKE) -C predictor quality-check
#	@$(MAKE) -C toxicology_testing quality-check
	@$(MAKE) -C face_recognizer_portfolio_milestone quality-check

run-test:
#	@$(MAKE) -C algebraic_predictor run-test
#	@$(MAKE) -C mnist_predictor run-test
#	@$(MAKE) -C predictor run-test
#	@$(MAKE) -C toxicology_testing run-test
	@$(MAKE) -C face_recognizer_portfolio_milestone run-test

format:
#	@$(MAKE) -C algebraic_predictor format
#	@$(MAKE) -C mnist_predictor format
#	@$(MAKE) -C predictor format
#	@$(MAKE) -C toxicology_testing format
	@$(MAKE) -C face_recognizer_portfolio_milestone format

run:
#	@$(MAKE) -C algebraic_predictor run
#	@$(MAKE) -C mnist_predictor run
#	@$(MAKE) -C predictor run
#	@$(MAKE) -C toxicology_testing run
	@$(MAKE) -C face_recognizer_portfolio_milestone run