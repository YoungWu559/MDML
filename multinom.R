require(nnet)
require(mlogit)
require(naivebayes)
require(e1071)

# seed = random seed
# dim = 1 or 2
# k = number of classes
# n = number of data
# sep = 0 (random label), 1 (linearly separable label)
# var = variance of line
gen_data <- function(seed = 0, dim = 2, k = 3, n = 100, sep = 1, var = 1)
{
  set.seed(seed)
  x <- matrix(rnorm(n * dim), n, dim)
  if (sep == 0) y <- as.factor(floor(runif(n) * k))
  if (sep == 1)
  {
    w <- matrix(rnorm((dim + 1) * k), dim + 1, k)
    z <- cbind(rep(1, n), x) %*% w + var * matrix(rnorm(n * k), n, k) 
    y <- as.factor(max.col(z, ties.method = "first"))
  }
  if (sep == 2)
  {
    nx <- as.vector(rmultinom(1, size = n - 4, prob = rep(1/4, 4))) + 1
    x <- c(rep(0, nx[1]), rep(1, nx[2]), rep(2, nx[3]), rep(-1, nx[4]))
    y <- c(rep(1, nx[1]), rep(2, nx[2]), rep(3, nx[3]), rep(2, nx[4]))
  }
  tab <- table(y)
  if (length(tab) != k || min(tab) <= 2) return(gen_data(seed + 1, dim, k, n, sep, var))
  if (dim == 1 || sep == 2) df <- data.frame(y = y, x1 = x)
  if (dim == 2) df <- data.frame(y = y, x1 = x[,1], x2 = x[,2])
  return(df)
}

# data = (y, x1, x2)
# method = 0 if multinom; 1 if bfgs, 1.1 if nm, 1.2 if gd; 2 if naive bayes
gen_model <- function(data, method = 0, lam = 0)
{
  if (method == 0)
  {
    if(ncol(data) == 2) ml <- multinom(y ~ x1, data = data, trace = FALSE)
    if(ncol(data) == 3) ml <- multinom(y ~ x1 + x2, data = data, trace = FALSE)
    return(ml)
  }
  if (floor(method) == 1)
  {
    w <- 0
    if (method == 1) w <- bfgs(data, lam)
    if (method == 1.1) w <- nm(data, lam)
    if (method == 1.2) w <- gd(data, lam)
    class(w) <- c("w", class(w))
    return(w)
  }
  if (method == 2) return(gaussian_naive_bayes(data.matrix(data[,2:ncol(data)]), data[,1]))
  if (method == 3) return(svm(y ~ x1 + x2, data = data, kernel = "linear", scale = FALSE, probability = TRUE))
  if (method == 3.1) return(svm(y ~ x1 + x2, data = data, kernel = "linear", scale = FALSE))
  if (method == 3.2) {
    nlevel <- length(levels(data$y))
    ws <- c()
    for (i in 1:nlevel)
    {
      data$yi <- as.factor((data$y == i) + 1)
      mi <- svm(yi ~ x1 + x2, data = data, kernel = "linear", scale = FALSE)
      beta <- t(mi$coefs) %*% as.matrix(data[mi$index,2:3])
      beta0 <- mi$rho
      ws <- rbind(ws, c(beta0 / beta[2], -beta[1] / beta[2], beta0, beta[1], beta[2]))
    }
    return(ws)
  }
}

print_model <- function(model, method = 0)
{
  if (method == 0) print(summary(model))
  if (floor(method) == 1)
  {
    table <- matrix(c(model[3], model[1], model[2], model[6], model[4], model[5]), nrow = 2, ncol = 3, byrow = T)
    rownames(table) <- c("2", "3")
    colnames(table) <- c("(Intercept)", "x1", "x2")
    print(table)
  }
  if (method == 2) print(summary(model))
  if (floor(method) == 3 && method != 3.2) print(summary(model))
  if (method == 3.2)
  {
    table <- matrix(model[,3:5], nrow = 3, ncol = 3)
    rownames(table) <- c("1", "2", "3")
    colnames(table) <- c("(Intercept)", "x1", "x2")
    print(table)
  }
}

predict.w <- function(object, newdata, ...)
{
  prob <- mle_activate(cbind(y = rep(0, nrow(newdata)),newdata), object[1:6])
  return(apply(prob, 1, function(x) which(x == max(x))))
}

gen_pred_new <- function(model, data, method = 0) {
  xc <- data.frame(x1 = data)
  return(predict(model, xc))
}
 
# model = either model or weight matrix
# data = (y, x1, x2)
# i = index
# yc = original label
gen_pred <- function(model, data, i = 1, yc = 0, method = 0)
{
  if (method == 0)
  {
    if (ncol(data) == 2) xc <- data.frame(x1 = data$x1[i])
    if (ncol(data) == 3) xc <- data.frame(x1 = data$x1[i], x2 = data$x2[i])
    k = nlevels(data$y)
    if (k == 2)
    {
      if (yc == 1) py <- predict(model, xc, "probs")[1]
      if (yc == 2) py <- 1 - predict(model, xc, "probs")[1]
    }
    if (k > 2) py <- predict(model, xc, "probs")[yc]
    return(py)
  }
  if (floor(method) == 1) return(mle_activate(data[i,], model)[yc])
  if (method == 2) return(predict(model, data.matrix(data[i,2:ncol(data)]), "prob")[yc])
  if (method == 3) return(attr(predict(model, data[i,], probability = TRUE), "probabilities")[yc])
  if (method == 3.1)
  {
    pred = predict(model, data[i,]);
    if (pred == yc) return(1);
    if (pred != yc) return(0);
  }
  if (method == 3.2) return((abs(data[i,2] * (-model[yc,1]) + data[i,3] + (-model[yc,2]))) / (sqrt(model[yc,1] * model[yc,1] + 1)))
}

test_svm <- function(data)
{
  mi <- gen_model(data, 3.2)
  plot(data$x2 ~ data$x1, col=data$y)
  abline(mi[1,1], mi[1,2])
}

gen_pred_all <- function(model, data, method = 0)
{
  if (method == 0)
  {
    if (ncol(data) == 2) xc <- data.frame(x1 = data$x1)
    if (ncol(data) == 3) xc <- data.frame(x1 = data$x1, x2 = data$x2)
    k = nlevels(data$y)
    if (k == 2) py <- c(predict(model, xc, "probs")[1], 1 - predict(model, xc, "probs")[1])
    if (k > 2) py <- predict(model, xc, "probs")
    return(py)
  }
  if (floor(method) == 1) return(mle_activate(data, model))
  if (method == 2) return(predict(model, data.matrix(data[,2:ncol(data)]), "prob"))
  if (method == 3) return(attr(predict(model, data, probability = TRUE), "probabilities"))
  if (method == 3.1) return(onehot(predict(model, data)))
  if (method == 3.2) {
    pred <- c()
    for (i in 1:dim(data)[1]) {
      pred <- rbind(pred, (abs(data[i,2] * (-model[,1]) + data[i,3] + (-model[,2]))) / (sqrt(model[,1] * model[,1] + 1)))
    }
    return(pred)
  }
}

print_table <- function (pred, predp, col = 3) {
  nr <- nrow(pred)
  names <- rep("", nr * col)
  out <- rep(0, nr * col)
  outp <- rep(0, nr * col)
  for (i in 1:nr) {
    for (j in 3:(2 + col)) {
      pre <- "P"
      if (j - 2 == pred[i, 2]) pre <- "P*"
      names[(i - 1) * col + j - 2] <- paste(pre, "(y=", j - 2, "|x=", pred[i, 1], ")", sep = "")
      out[(i - 1) * col + j - 2] <- pred[i, j]
      outp[(i - 1) * col + j - 2] <- predp[i, j]
    }
  }
  combined <- rbind(out, outp)
  colnames(combined) <- names
  rownames(combined) <- c("Original", "Flipped")
  return(combined)
}

# data = (y, x1, x2)
# seed = seed
# out = whether to output
# lam = regularization
test <- function (data, seed = 0, out = FALSE, method = 0, lam = 0)
{
  dim <- 1
  ml <- gen_model(data, method, lam)
  ic <- 0
  sym <- rep(20, nrow(data))
  if (out) 
  {
    print("Original Model")
    print_model(ml, method)
    pred <- cbind(data$x1, data$y, gen_pred_all(ml, data, method))
    pred <- unique(pred)
    colnames(pred) <- c("x", "y", 1:(ncol(pred)-2))
    rownames(pred) <- 1:nrow(pred)
    print(round(pred, 4))
  }
  for (i in 1:nrow(data))
  {
    yi <- data$y[i]
    py <- gen_pred(ml, data, i, yi, method)
    for (j in 1:nlevels(data$y))
    {
      note <- ""
      data$y[i] <- j
      mlj <- gen_model(data, method, lam)
      pyj <- gen_pred(mlj, data, i, yi, method)
      if (out && i == 1 && j == 3) {
        data$y[i] <- yi
        predp <- cbind(data$x1, data$y, gen_pred_all(mlj, data, method))
        predp <- unique(predp)
      }
      if (pyj > py + 0.001) 
      {
        note <- "<- not IC"
        sym[i] <- 15
        ic <- ic + 1
        defend <- 0
        if (seed >= 0) defend <- test(data, -1, FALSE, method, lam);
        if (defend > 0) print("can defend");
        if (out) 
        {
          print(paste("New Model if ", i, " reports ", j, " instead of ", yi))
          print_model(mlj, method)
          data$y[i] <- yi
          predp <- cbind(data$x1, data$y, gen_pred_all(mlj, data, method))
          predp <- unique(predp)
          colnames(predp) <- c("x", "y", 1:(ncol(pred)-2))
          rownames(predp) <- 1:nrow(pred)
          print(round(predp, 4))
          if (out) dplot(mlj, data, sym)
        }
      }
      if (out) print(paste("y =", yi, ", p(y) =", round(py, 4), ", y' =", j, ", p'(y) =", round(pyj, 4), note))
    }
    data$y[i] <- yi
  }
  if (ic > 0) print(paste(seed, " not IC * ", ic))
  if (out) dplot(ml, data, sym)
  if (out) print(data)
  if (out) print(round(print_table(pred, predp, 3), 4))
  return(ic)
}

# data = (y, x1, x2)
# weights = (0; w1; w2)
loglike <- function(data, weights, lam = 0)
{
  ps <- mle_activate(data, weights)
  p <- c()
  for (i in 1:nrow(data)) p <- c(p, ps[i,data$y[i]])
  return(-sum(log(p)) / nrow(data) - lam * sqrt(sum(weights^2)))
}

# data = (y, x1, x2)
# weights = (0; w1; w2)
dloglike <- function(data, weights, lam = 0)
{
  if (ncol(data) == 3) design <- cbind(data$x1, data$x2, rep(1, nrow(data)))
  if (ncol(data) == 2) design <- cbind(data$x1, rep(1, nrow(data)))                
  ps <- mle_activate(data, weights)
  db <- onehot(data$y) - ps
  dw <- matrix(0, nrow = length(levels(data$y)), ncol = ncol(data))
  for(i in 1:nrow(data)) dw <- dw + db[i,] %o% design[i,]
  grad <- as.vector(t(dw[-1,])) + lam * weights
  return(-grad)
}

# y
onehot <- function(y)
{
  n <- length(y)
  m <- length(levels(y))
  one <- matrix(0, nrow = n, ncol = m)
  for(i in 1:n) one[i,y[i]] = 1
  return(one)
}

# data = (y, x1, x2)
# weights = (0; w1; w2)
mle_activate <- function(data, weights)
{
  if (ncol(data) == 3) design <- cbind(data$x1, data$x2, rep(1, nrow(data)))
  if (ncol(data) == 2) design <- cbind(data$x1, rep(1, nrow(data)))                
  w <- cbind(rep(0, ncol(data)), matrix(weights, ncol(data), length(weights) / ncol(data)))
  z <- design %*% w
  #ez <- exp(z)
  ps <- ez / rowSums(ez)
  return(ps)
}

# data = (y, x1, x2)
gd <- function(data, lam = 0)
{
  weights <- rep(0, ncol(data) * (nlevels(data$y) - 1))
  loss <- loglike(data, weights, lam)
  nloss <- loss / 2
  niter <- 0
  while((loss - nloss) / loss > 0.00001 & niter < 100000)
  {
    weights <- weights - 1 / (0.25 * (9 + 1)) * dloglike(data, weights, lam)
    niter <- niter + 1
    loss <- nloss
    nloss <- loglike(data, weights, lam)
  }
  return(weights)
}

# data = (y, x1, x2)
nm <- function(data, lam = 0)
{
  mleloglike <- function(weights) loglike(data, weights, lam);
  weights <- optim(rep(0, ncol(data) * (nlevels(data$y) - 1)), mleloglike)
  return(weights$par)
}

# data = (y, x1, x2)
bfgs <- function(data, lam = 0)
{
  bfgsloglike <- function(weights) loglike(data, weights, lam);
  bfgsdloglike <- function(weights) dloglike(data, weights, lam);
  weights <- optim(rep(0, ncol(data) * (nlevels(data$y) - 1)), bfgsloglike, bfgsdloglike, method = "BFGS")
  return(weights$par)
}

decisionplot <- function(model, data, class = NULL, sym = NULL, predict_type = "class",
                         resolution = 100, showgrid = TRUE, ...) {
  
  if(!is.null(class)) cl <- data[,class] else cl <- 1
  data <- data[,1:2]
  k <- length(unique(cl))

  plot(data, col = as.integer(cl)+1L, pch = sym, cex = 2, ...)
  #plot(data, col = as.integer(cl)+1L, pch = as.integer(cl)+1L, ...)
  
  # make grid
  r <- sapply(data, range, na.rm = TRUE)
  xs <- seq(r[1,1], r[2,1], length.out = resolution)
  ys <- seq(r[1,2], r[2,2], length.out = resolution)
  g <- cbind(rep(xs, each=resolution), rep(ys, time = resolution))
  colnames(g) <- colnames(r)
  g <- as.data.frame(g)
  #g <- data.matrix(g)
  
  ### guess how to get class labels from predict
  ### (unfortunately not very consistent between models)
  p <- predict(model, g, type = predict_type)
  if(is.list(p)) p <- p$class
  p <- as.factor(p)
  
  if(showgrid) points(g, col = as.integer(p)+1L, pch = ".")
  z <- matrix(as.integer(p), nrow = resolution, byrow = TRUE)
  contour(xs, ys, z, add = TRUE, drawlabels = FALSE,
          lwd = 2, levels = (1:(k-1))+.5)
  
  invisible(z)
}

dplot <- function(model, data, sym)
{
  if (ncol(data) == 2)
  {
    plot(as.numeric(y) ~ x1, data = data, col = c("red", "green", "blue")[y], pch = sym, xlim = c(-2, 2), ylim = c(0, nlevels(data$y) + 1))
    for(i in seq(-1, 1, by = 0.01)) points(i, 0, col = c("red", "green", "blue")[gen_pred_new(model, i, 0)])
  }
  if (ncol(data) == 3)
  {
    ndata <- cbind(data[,2:ncol(data)], y = as.numeric(data[,1]) * 1.0)
    decisionplot(model, ndata, "y", sym)
  }
}

# n number of iterations
repeat_test <- function()
{
  total <- 0
  for (s in 7000:10000)
  {
    if(s %% 100 == 0) print(s)
    data <- gen_data(seed = s, dim = 2, k = 3, n = 100, sep = 0, var = 0.01)
    ic <- test(data, seed = s, out = FALSE, method = 0, lam = 0.1)
    if (ic > 0) total <- total + 1
  }
  print(total)
}

repeat_test()

#data <- gen_data(seed = 217, dim = 2, k = 3, n = 20, sep = 1, var = 1)
#test(data, seed = 217, out = FALSE, method = 1, lam = 0.01)

circular_data <- function(r = 1, n = 10)
{
  x1 <- r * sin(1:(3 * n) / (3 * n) * 2 * pi)
  x2 <- r * cos(1:(3 * n) / (3 * n) * 2 * pi)
  y <- as.factor(c(rep(1, n), rep(2, n), rep(3, n)))
  return(data.frame(y = y, x1 = x1, x2 = x2))
}

boundary_data <- function(r = 1, n = 10, off = 0.1)
{
  theta <- c(0, 2 / 3 * pi, 4 / 3 * pi)
  x1 <- c((1:n) / n * r * sin(theta[1]), (1:n) / n * r * sin(theta[2]), (1:n) / n * r * sin(theta[3]))
  x1 <- c(x1, (1:n) / n * r * sin(theta[1] + off), (1:n) / n * r * sin(theta[2] + off), (1:n) / n * r * sin(theta[3] + off))
  x2 <- c((1:n) / n * r * cos(theta[1]), (1:n) / n * r * cos(theta[2]), (1:n) / n * r * cos(theta[3]))
  x2 <- c(x2, (1:n) / n * r * cos(theta[1] + off), (1:n) / n * r * cos(theta[2] + off), (1:n) / n * r * cos(theta[3] + off))
  y <- as.factor(c(rep(1, n), rep(3, n), rep(2, n), rep(3, n), rep(2, n), rep(1, n)))
  return(data.frame(y = y, x1 = x1, x2 = x2))
}

offset_boundary_data <- function(r = 1, n = 10, off = 0.05, rot = 0)
{
  #theta <- c(0, 2 / 3 * pi, 4 / 3 * pi)
  theta <- c(0, 2 / 3 * pi, 4 / 3 * pi) + rot
  ax <- off * sin(pi / 2 + theta)
  ay <- off * cos(pi / 2 + theta)
  x2 <- c((1:n) / n * r * sin(theta[1]) + ax[1], (1:n) / n * r * sin(theta[2]) + ax[2], (1:n) / n * r * sin(theta[3]) + ax[3])
  x2 <- c(x2, (1:n) / n * r * sin(theta[1]) - ax[1], (1:n) / n * r * sin(theta[2]) - ax[2], (1:n) / n * r * sin(theta[3]) - ax[3])
  x1 <- c((1:n) / n * r * cos(theta[1]) + ay[1], (1:n) / n * r * cos(theta[2]) + ay[2], (1:n) / n * r * cos(theta[3]) + ay[3])
  x1 <- c(x1, (1:n) / n * r * cos(theta[1]) - ay[1], (1:n) / n * r * cos(theta[2]) - ay[2], (1:n) / n * r * cos(theta[3]) - ay[3])
  y <- as.factor(c(rep(1, n), rep(2, n), rep(3, n), rep(3, n), rep(1, n), rep(2, n)))
  return(data.frame(y = y, x1 = x1, x2 = x2))
}

# type = 0 circular, 1 boundary, 2 offset boundary
test_special <- function(n = 10, off = 0.05, type = 0)
{
  count = 6
  if(type == 0) count = 3
  if (type == 0) data <- circular_data(1, n)
  if (type == 1) data <- boundary_data(1, n, off)
  if (type == 2) data <- offset_boundary_data(1, n, off)
  test(data, seed = 0, out = FALSE, method = 0, lam = 0.01)
  for(t in 1:(count * n))
  {
    if (type == 0) data <- circular_data(1, n)
    if (type == 1) data <- boundary_data(1, n, off)
    if (type == 2) data <- offset_boundary_data(1, n, off)
    data[t,1] <- 1
    print(t)
    test(data, seed = 0, out = FALSE, method = 0, lam = 0.01)
  }
}

one_d_test <- function(n = c(5, 5, 5), y = c(1), x = c(-1)) 
{
  return(data.frame(y = as.factor(c(y, rep(1, n[1]), rep(2, n[2]), rep(3, n[3]))), x1 = c(x, rep(-1, n[1]), rep(0, n[2]), rep(1, n[3]))))
}

one_d_test_t <- function(n = c(5, 5, 5, 5, 5), ep = 0.01, y = c(1), x = c(-1))
{
  return(data.frame(y = as.factor(c(y, rep(1, n[1]), rep(2, n[2]), rep(1, n[3]), rep(2, n[4]), rep(3, n[5]))), x1 = c(x, rep(-1, n[1]), rep(-0.5 - ep, n[2]), rep(-0.5 + ep, n[3]), rep(0, n[4]), rep(1, n[5]))))
}

one_d_test_m <- function(x0, y0)
{
  return(data.frame(y = as.factor(y0), x1 = x0))
}

#test(one_d_test_m(c(-0.53, -0.51, -0.51, -0.51, -0.49, -0.49, -0.49, 1, 1, 1), c(2, 1, 1, 1, 2, 2, 2, 3, 3, 3)), 1, TRUE)
#test(one_d_test_m(c(-0.53, -0.53, -0.51, -0.51, -0.51, -0.49, -0.49, 1, 1, 1), c(2, 2, 1, 1, 1, 2, 2, 3, 3, 3)), 0, TRUE)
#test(one_d_test_m(c(-0.53, -0.51, -0.51, -0.49, -0.49, 1, 1, 1), c(2, 1, 1, 2, 2, 3, 3, 3)), 0, TRUE)
#test(one_d_test_m(c(-0.53, -0.51, -0.51, -0.49, 1, 1, 1), c(2, 1, 1, 2, 3, 3, 3)), 0, TRUE)
#test(one_d_test_m(c(-0.53, -0.51, -0.51, -0.51, -0.49, -0.49, -0.49, 1, 1, 1) * 2, c(2, 1, 1, 1, 2, 2, 2, 3, 3, 3)), 0, TRUE)
#test(one_d_test_m(c(-0.53, -0.51, -0.51, -0.51, -0.49, -0.49, -0.49, 1, 1, 1) * 0.5, c(2, 1, 1, 1, 2, 2, 2, 3, 3, 3)), 0, TRUE)
#test(one_d_test_m(c(-0.53, -0.51, -0.51, -0.51, -0.49, -0.49, -0.49, 1, 1, 1) + 1, c(2, 1, 1, 1, 2, 2, 2, 3, 3, 3)), 0, TRUE)
#test(one_d_test_t(c(0, 3, 3, 0, 3), -0.01, c(2), c(-0.53)), 0, TRUE)
#test(one_d_test_t(c(0, 4, 3, 0, 3), -0.01, c(2), c(-0.51)), 0, TRUE)

#test_special(3, 0.01, 2)
#data <- offset_boundary_data(1, 3, 0.004, pi * 0.5)
#data[5,1] <- 1
#test(data, seed = 0, out = TRUE, method = 0, lam = 0.01)

#data <- offset_boundary_data(1, 3, 0.01, 0)
#data <- rbind(data, c(y = 1, x1 = 0.75, y1 = -0.01))
#test(data, seed = y, out = TRUE, method = 2, lam = 0.01)

#data <- offset_boundary_data(1, 3, 0.004, pi * 0.5)
#data[5,1] <- 1
#test(data, seed = 0, out = TRUE, method = 1.2, lam = 0.01)

#data <- circular_data(1, 11)
#data[15,1] <- 1
#test(data, seed = 0, out = TRUE, method = 0, lam = 1)
#data <- boundary_data(1, 3)
#data[8,1] <- 1
#test(data, seed = 0, out = TRUE, method = 0, lam = 0.1)

#data <- gen_data(4, 2, 3, 20, 1, 1)
#test(data, seed = 4, out = TRUE, method = 2)

#data <- gen_data(2, 2, 3, 20, 1, 1)
#test(data, seed = 2, out = TRUE, method = 0)
#test(data, seed = 2, out = TRUE, method = 1, lam = 0.01)
#test(data, seed = 2, out = TRUE, method = 1, lam = 0.1)
#test(data, seed = 2, out = TRUE, method = 1, lam = 0)

#data <- gen_data(217, 2, 3, 20, 1, 1)
#test(data, seed = 0, out = TRUE)

#data <- data.frame(y = as.factor(c(3, 2, 2, 1)), x1 = c(0, 1, 2, 3))
#data <- data.frame(y = as.factor(c(1, 2, 3, 3)), x1 = c(0.1, 1.1, -1.1, -1.5), x2 = c(0.5, 2.3, -2.3, -2.5))

#test2d(seed = 1805, out = TRUE, k = 3, n = 20, var = 0, sep = 1)
#test2d(seed = 217, out = TRUE, k = 3, n = 20, var = 1, sep = 1)
#test1d(seed = 585, out = TRUE, k = 3, n = 20, var = 0, sep = 1)
#test1d(seed = 421, out = TRUE, k = 3, n = 20, var = 1, sep = 1)
#test1d(seed = 0, out = TRUE, k = 3, n = 20, var = 0, sep = 1)

# x1 <- rnorm(1000)
# x2 <- x1 + 100
# xp <- x1 + rnorm(1000, 0, 0.1)
# y <- as.factor(xp > 0)
# m1 <- multinom(y ~ x1)
# m2 <- multinom(y ~ x2)
# p1 <- predict(m1, type = "prob")
# p2 <- predict(m2, type = "prob")
# round(cbind(p1, p2), 2)

