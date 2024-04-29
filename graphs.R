method <- c("Pure CPU", "ant_solution", "ant_solution_coarse")
time_s <- c(0.37348, 0.08953, 0.08732)

png('method_bargraph.png')

barplot(time_s, names.arg=method, xlab="Method", ylab="Execution Time (s)", main="Execution time per method", col="skyblue", width=0.6, ylim=c(0,max(time_s)*1.2), bg="lightgray")

dev.off()
