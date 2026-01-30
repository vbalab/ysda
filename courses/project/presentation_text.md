1. The problem of time series modeling using diffusion models, specifically, try to come up with enchancements to existing models.

2. For that I've read a lot of literature touching on applying diffusion models for time series forecasting. Additionally, I focused on models that were realeased mid 2025 touching on applying Flow Matching to vector time-series forecasting.

3. While looking through literature I became interested in interpretable time-series forecasting, which was beautifully done in DiffusionTS framework. Later I found FMTS model that applied DiffusionTS architecture to Flow Matching way of sampling. They both mainly focused on imputation doing implicit conditioning via replacements of & guidence to observed values, so no direct conditional (for forecasting task including) training was done. In this backbone transformer architecture time-series imputation was interpretable to 3 components: trend, seasonal, error. Also these models were doing unconventional way of doing time steps training on original $x$ prediction rather that traditional way of predicting noise $\epsilon$ added at the step.

4. While studing the literature I was a bit confused, since there was no framework which did any sort of preprocessing or training in a form of taking difference on time-series data, which is crusial thing for almost any time-series data, which is mostly first order integrated and, consiquently, non-stationary.

5. I wanted to apply traditional time-series econometric modelling methods to SoTA Flow Matching method of modelling, while keeping it interpretable.

6. For that I came up with improvements to the FMTS framework, while being conserned specifically with forecasting task solely. For that I've restructure a model for forecasting task specifically, so that it was trained conditionally on history, made it use first differenced stationary data and added jump component that is viewed as a traditional element in financial time-series econometrics. For a specific representation of jump component, I choosed self-exciting jumps (Hawkes-style) with explicit history.

7. I've developed a (full framework)[https://github.com/vbalab/FITS] that includes multiple forecasting model families (FITS as my model, and also CSDI, DiffusionTS, FMTS used through _adapters_), dataset helpers, and training/evaluation utilities.

8. I used two datasets that were considered to be conventional for time series forecasting task, ETTh and Solar Energy. Trained and evaluated CSDI, DiffusionTS and FMTS models in order to compare them with FITS (mine) through various metrics (RMSE, MAE, CRPS and CRPS sum) and visualization tools (data density, PCA, t-SNE).

9. As a result it appears that making data stationary alone makes the task of forecasting significantly easier (and even has greater effect that conditional learning), while jump component while not only adding more interpretability to the results, makes better forecasting itself while not making model significantly heavier.

10. The job is _not done_ in a full measure, since I'm planning on continuing this project as my degree work at New Economic School. I already see that model encoder part might be better with CSDI's way of doing mixed time|feature learning (which blows GPU memory by a bit, but as I see it might have better potential), while keeping DiffusionTS's decoder that does component decomposition into interpretable elements.
