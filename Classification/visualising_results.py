#-------------------- Visualising the Results --------------------------------

#xrisimopoiw to kalutero modelo gia na kanw predict ton deikti pm10
compare_data = pd.DataFrame({'dates':df['datetime'],
                            'Actual PM10':y,
                            'Predicted PM10':best_model1.predict(X)})
compare_data["Actual PM10"] = pd.to_numeric(compare_data["Actual PM10"])
compare_data["Predicted PM10"] = pd.to_numeric(compare_data["Predicted PM10"])
compare_data.set_index('dates',inplace=True)
compare_data['Predicted PM10'] = np.round(compare_data['Predicted PM10'],1)


# Plot Figures xwris compare
fignow = plt.figure(figsize=(8,8))

x = compare_data['Actual PM10']
y = compare_data["Predicted PM10"]

## find the boundaries of X and Y values
bounds = (min(x.min(), y.min()) - int(0.1 * y.min()), max(x.max(), y.max())+ int(0.1 * y.max()))

# Reset the limits
ax = plt.gca()
ax.set_xlim(bounds)
ax.set_ylim(bounds)
# Ensure the aspect ratio is square
ax.set_aspect("equal", adjustable="box")

plt.plot(x,y,"o", alpha=0.5 ,ms=10, markeredgewidth=0.0)

ax.plot([0, 1], [0, 1], "r-",lw=2 ,transform=ax.transAxes)

# Calculate Statistics of the Parity Plot 
mean_abs_err = np.mean(np.abs(x-y))
rmse = np.sqrt(np.mean((x-y)**2))
rmse_std = rmse / np.std(y)
#z = np.polyfit(x,y, 1)
#y_hat = np.poly1d(z)(x)
# Title and labels 
plt.title("Parity Plot")
plt.xlabel('Actual')
plt.ylabel('Predicted')


# plot the daily averages of the Actual PM10 and the predicted PM10
compare_data = compare_data.resample('D').mean()

with plt.style.context('fivethirtyeight'):
    plt.figure(figsize=(12,5))
    plt.scatter(compare_data.index,compare_data['Actual PM10'],s=15,label='Actual PM10',
               alpha=.6)
    plt.scatter(compare_data.index,compare_data['Predicted PM10'],s=15,label='Predicted PM10',
               alpha=.6)
    plt.legend()
    plt.title('Evaluating the model\n',
             fontsize=18)
    plt.xlabel('period',fontsize=15)
    plt.ylabel('PM10 concentration',fontsize=15)
    plt.show()  


# Plot Figures me compare 
fignow = plt.figure(figsize=(8,8))

x = compare_data['Actual PM10']
y = compare_data["Predicted PM10"]

## find the boundaries of X and Y values
bounds = (min(x.min(), y.min()) - int(0.1 * y.min()), max(x.max(), y.max())+ int(0.1 * y.max()))

# Reset the limits
ax = plt.gca()
ax.set_xlim(bounds)
ax.set_ylim(bounds)
# Ensure the aspect ratio is square
ax.set_aspect("equal", adjustable="box")

plt.plot(x,y,"o", alpha=0.5 ,ms=10, markeredgewidth=0.0)

ax.plot([0, 1], [0, 1], "r-",lw=2 ,transform=ax.transAxes)

# Calculate Statistics of the Parity Plot 
mean_abs_err = np.mean(np.abs(x-y))
rmse = np.sqrt(np.mean((x-y)**2))
rmse_std = rmse / np.std(y)
#z = np.polyfit(x,y, 1)
#y_hat = np.poly1d(z)(x)
# Title and labels 
plt.title("Parity Plot")
plt.xlabel('Actual')
plt.ylabel('Predicted')