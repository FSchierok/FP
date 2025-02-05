# http://www.scikit-yb.org/en/latest/api/features/importances.html
from sklearn.ensemble import GradientBoostingClassifier
from yellowbrick.features.importances import FeatureImportances

gbc = GradientBoostingClassifier()
gbc.fit(X,y)
importances = gbc.feature_importances_
indices = np.argsort(importances)[::-1]

# Print the Top 20 feature ranking
print("Feature ranking:")
for f in range(20):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

# Plot the 20 largest feature importances
fig0 = plt.figure(0)
ax0 = fig0.add_subplot(111)
ax0.set_title("Die 20 wichtigsten Features")
ax0.bar(range(20), importances[indices[0:20]], align="center")
ax0.set_ylabel("importance")
ax0.set_xticks(range(20))
ax0.set_xticklabels(indices[0:20])
fig0.savefig("plots/featureImportance.pdf")
