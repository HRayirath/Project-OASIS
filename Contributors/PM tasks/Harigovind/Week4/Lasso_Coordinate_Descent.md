Lasso Regression using Coordinate Descent
Unlike Ridge Regression, Lasso Regression does not have a closed-form analytical solution because of the presence of the L₁ regularization term. The objective function minimized by Lasso is
J(β) = ‖y − Xβ‖² + λ ∑ⱼ₌₁ᵖ |βⱼ|.
The absolute value term is not differentiable at zero, making it impossible to obtain the coefficients by directly solving a system of linear equations. Instead, the coefficients are estimated iteratively using the Coordinate Descent algorithm.
During each iteration, one coefficient is updated while keeping all the remaining coefficients fixed. Suppose the algorithm is currently updating the coefficient βⱼ. The contribution of this feature is temporarily removed from the model prediction, and the residual is computed as
r = y − (Xβ − xⱼβⱼ),
where xⱼ denotes the jᵗʰ feature column. This residual represents the portion of the target that is still unexplained by all the other features, allowing the current coefficient to be optimized independently.
To simplify the optimization, two quantities are introduced:
ρ = xⱼᵀr,
which measures how strongly the current feature is correlated with the remaining residual, and
z = xⱼᵀxⱼ,
which is simply the squared Euclidean norm of the feature column.
Substituting these into the objective reduces the optimization problem for a single coefficient to
J(βⱼ) = zβⱼ² − 2ρβⱼ + λ|βⱼ|.
Since the absolute value function is not differentiable at zero, the optimization is solved piecewise by considering three cases:
If βⱼ > 0, the optimal update is

 βⱼ = (2ρ − λ)/(2z).


If βⱼ < 0, the optimal update is

 βⱼ = (2ρ + λ)/(2z).


If neither condition is satisfied, the minimum occurs at

 βⱼ = 0.


These three cases together form the soft-thresholding operator, which is the defining feature of Lasso Regression. Unlike Ridge Regression, which merely shrinks coefficients towards zero, Lasso can set coefficients exactly equal to zero. This property enables automatic feature selection, allowing insignificant predictors to be removed from the model while retaining the important ones.
The coordinate descent process repeatedly updates each coefficient using the above rule until the coefficient vector converges. Convergence is determined by comparing the coefficient vector before and after an iteration. If the Euclidean norm of the change falls below a predefined tolerance, the algorithm terminates. This iterative procedure produces the final sparse coefficient vector used for prediction.

