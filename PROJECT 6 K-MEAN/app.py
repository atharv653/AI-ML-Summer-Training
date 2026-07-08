import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris

st.set_page_config(page_title="K-Means Clustering", page_icon="🌸", layout="wide")
st.title("🌸 K-Means Clustering using Iris Dataset")

iris=load_iris()
df=pd.DataFrame(iris.data,columns=iris.feature_names)
st.subheader("Dataset")
st.dataframe(df)

X=df[['petal length (cm)','petal width (cm)']]
k=st.sidebar.slider("Number of Clusters",2,8,3)

model=KMeans(n_clusters=k,random_state=42,n_init=10)
df["Cluster"]=model.fit_predict(X)

st.subheader("Clustered Dataset")
st.dataframe(df)

fig,ax=plt.subplots()
colors=["red","green","blue","orange","purple","brown","pink","black"]
for i in range(k):
    t=df[df["Cluster"]==i]
    ax.scatter(t["petal length (cm)"],t["petal width (cm)"],color=colors[i],label=f"Cluster {i}")
c=model.cluster_centers_
ax.scatter(c[:,0],c[:,1],marker="*",s=250,color="black",label="Centroids")
ax.legend()
st.pyplot(fig)

sse=[]
for i in range(1,10):
    km=KMeans(n_clusters=i,random_state=42,n_init=10).fit(X)
    sse.append(km.inertia_)
fig2,ax2=plt.subplots()
ax2.plot(range(1,10),sse,marker="o")
ax2.set_title("Elbow Method")
st.pyplot(fig2)
