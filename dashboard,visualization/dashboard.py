import streamlit as st
import pandas as pd
import numpy as np
import folium
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px


st.set_page_config(page_title="부동산 대시보드", layout="wide", initial_sidebar_state="expanded")

# --------------------------
# 1) 데이터 로드
# --------------------------
@st.cache_data
def load_data():
    raw_unique = pd.read_csv("2018_2024_지오코딩안되어있는결측치제거.csv", encoding='utf-8')
    raw_full = pd.read_csv("2018_2024_결측치 제거된 로우데이터.csv", encoding='utf-8')
    art = pd.read_csv("art_data.csv", encoding='utf-8')
    school = pd.read_csv("school_data.csv", encoding='utf-8')
    subway = pd.read_csv("subway.csv", encoding='utf-8')
    park = pd.read_csv("park_data.csv", encoding='utf-8')
    bus_stop = pd.read_csv("bus_stop_data.csv", encoding='utf-8')
    hospital = pd.read_csv("hospital_data.csv", encoding='euc-kr')
    big_market = pd.read_csv("big_markettt.csv", encoding='utf-8')
    people = pd.read_csv("people.csv", encoding="utf-8")
    household = pd.read_csv("house_hold.csv", encoding="utf-8")
    crime = pd.read_csv("final_crime.csv").drop(columns=["Unnamed: 0"], errors='ignore')
    return raw_unique, raw_full, art, school, subway, park, bus_stop, hospital, big_market, people, household, crime

(raw_unique, raw_full, art, school, subway, park, 
 bus_stop, hospital, big_market, people, household, crime) = load_data()

# --------------------------
# 2) 사이드바: 주택 필터
# --------------------------
#st.sidebar.header("🏠 주택 필터")
st.sidebar.header("🏠 주택 필터")

# 가격 (만원)
price_min_default = 50000   # 시작 시 최소값 예시
price_max_default = 100000  # 시작 시 최대값 예시
price_min = st.sidebar.number_input(
    "💰 물건금액 최소(만원)",
    int(raw_full["물건금액(만원)"].min()),
    int(raw_full["물건금액(만원)"].max()),
    price_min_default,
    step=1000
)
price_max = st.sidebar.number_input(
    "💰 물건금액 최대(만원)",
    int(raw_full["물건금액(만원)"].min()),
    int(raw_full["물건금액(만원)"].max()),
    price_max_default,
    step=1000
)

# 평수 (평) → 모두 float로 통일
py_min_default = 20.0
py_max_default = 30.0
py_min = st.sidebar.number_input(
    "📏 평수 최소(평)",
    float(raw_full["평수(평)"].min()),
    float(raw_full["평수(평)"].max()),
    py_min_default,
    step=1.0
)
py_max = st.sidebar.number_input(
    "📏 평수 최대(평)",
    float(raw_full["평수(평)"].min()),
    float(raw_full["평수(평)"].max()),
    py_max_default,
    step=1.0
)

# 건축년도 → int 유지
year_min_default = 2000
year_max_default = 2024
year_min = st.sidebar.number_input(
    "🏗 건축년도 최소",
    int(raw_full["건축년도"].min()),
    int(raw_full["건축년도"].max()),
    year_min_default
)
year_max = st.sidebar.number_input(
    "🏗 건축년도 최대",
    int(raw_full["건축년도"].min()),
    int(raw_full["건축년도"].max()),
    year_max_default
)
# --------------------------
# 3) 기본 필터 적용 (테이블1)
# --------------------------
basic_filtered = raw_full[
    (raw_full["물건금액(만원)"] >= price_min) &
    (raw_full["물건금액(만원)"] <= price_max) &
    (raw_full["평수(평)"] >= py_min) &
    (raw_full["평수(평)"] <= py_max) &
    (raw_full["건축년도"] >= year_min) &
    (raw_full["건축년도"] <= year_max)
].copy()



# --------------------------
# 4) 인프라 필터
# --------------------------
st.sidebar.header("🏢 인프라 필터")
infra_info = {
    "문화시설": art, "학교": school, "지하철": subway,
    "공원": park, "버스정류장": bus_stop, "병원": hospital, "대형마트": big_market
}
selected_infra = {}
infra_distance = {}
for name, df in infra_info.items():
    checked = st.sidebar.checkbox(name, False)
    selected_infra[name] = checked
    if checked:
        distance = st.sidebar.number_input(f"{name} 거리 기준 (m)", 100, 5000, 500, step=100)
        infra_distance[name] = distance
    else:
        infra_distance[name] = None

def haversine(lat1, lon1, lat2, lon2):
    R = 6371000
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    delta_phi = np.radians(lat2 - lat1)
    delta_lambda = np.radians(lon2 - lon1)
    a = np.sin(delta_phi/2.0)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(delta_lambda/2.0)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    return R * c


all_columns = raw_full.columns.tolist()
selected_columns = st.sidebar.multiselect("📝 표시할 컬럼 선택", options=all_columns, default=all_columns)

# --------------------------
# 5) 인프라 필터 적용 (테이블2)
# --------------------------
infra_filtered = basic_filtered.copy()
if any(selected_infra.values()):
    filtered_indices = set(infra_filtered.index.tolist())
    for name, checked in selected_infra.items():
        if checked:
            infra_df = infra_info[name]
            distance_limit = infra_distance[name]
            in_range_indices = set()
            for _, infra_row in infra_df.iterrows():
                dists = haversine(
                    infra_row['lat'], infra_row['lng'],
                    infra_filtered['lat'].values,
                    infra_filtered['lng'].values
                )
                in_range_indices.update(np.where(dists <= distance_limit)[0])
            filtered_indices &= in_range_indices
    infra_filtered = infra_filtered.iloc[list(filtered_indices)]
infra_filtered = infra_filtered.reset_index(drop=True)
infra_filtered.index += 1

# --------------------------
# 6) Streamlit 레이아웃: col1 전체 사용
# --------------------------
col1 = st.columns([1])[0]  # col1 하나로 전체 사용
col2, col3 = st.columns([1,1])  # col3부터 기존 col3,col4 역할

# --- 테이블1 ---
with col1:
    st.subheader(f"주택 필터 적용 ({len(basic_filtered)}건)")
    if len(basic_filtered) > 0:
        st.dataframe(basic_filtered[selected_columns])
    else:
        st.info("필터 조건에 맞는 데이터가 없습니다.")

# --- 테이블2 ---
with col2:
    st.subheader(f"인프라 필터 적용 ({len(infra_filtered)}건)")
    if len(infra_filtered) > 0:
        st.dataframe(infra_filtered[selected_columns])
    else:
        st.info("인프라 필터 조건에 맞는 데이터가 없습니다.")

# --- 지도2 ---
with col3:
    st.subheader("주택 및 인프라 필터 기준 매물")
    if len(infra_filtered) > 0:
        map_df2 = raw_unique[raw_unique["주소"].isin(infra_filtered["주소"])].copy()
        # 지도 중심 항상 서울
        center_lat, center_lng = 37.5665, 126.9780
        m2 = folium.Map(location=[center_lat, center_lng], zoom_start=11)
        for _, row in map_df2.iterrows():
            folium.Marker([row["lat"], row["lng"]], popup=row.get("건물명_x","")).add_to(m2)
        m2.save("map2.html")
        with open("map2.html","r",encoding="utf-8") as f:
            st.components.v1.html(f.read(), height=400, scrolling=True)
    else:
        st.info("인프라 필터 조건에 맞는 지도 데이터가 없습니다.")

# --------------------------
# 7) infra_filtered 기준 Top7
# --------------------------
if len(infra_filtered) > 0:
    final_result = infra_filtered.copy()
    final_result = final_result.sort_values(
        by=["계약 연도","평단가(만원)","평수(평)"],
        ascending=[False,True,False]
    ).drop_duplicates(subset="주소").head(5)
    st.subheader("Top5 매물 (평단가 낮고 평수 큰 매물)")
    st.dataframe(final_result[selected_columns])
else:
    st.info("Top5 매물 데이터가 없습니다.")

# --------------------------
# 한글 폰트 설정
# --------------------------
plt.rc('font', family='Apple SD Gothic Neo')
plt.rc('axes', unicode_minus=False)

# --------------------------
# final_result 기준 자치구만 선택
# --------------------------
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd





# 한글 폰트 설정
plt.rc('font', family='Apple SD Gothic Neo')
plt.rc('axes', unicode_minus=False)

# --------------------------
# 1) 인구 데이터 불러오기
# --------------------------
people = pd.read_csv("people.csv", encoding="utf-8")

# --------------------------
# 2) final_result 기준 자치구 필터링
# --------------------------
filtered_districts = final_result["자치구명"].unique()
people_filtered = people[people["자치구명"].isin(filtered_districts)]

# --------------------------
# 3) Wide -> Long 변환
# --------------------------
people_long = pd.melt(
    people_filtered,
    id_vars=['자치구명'],
    value_vars=['청년층','중장년층','노년층'],
    var_name='연령대',
    value_name='인구수'
)

# --------------------------
# 4) 바그래프 그리기
# --------------------------

st.subheader("자치구별 연령대별 인구수")

plt.figure(figsize=(16,6))
ax = sns.barplot(
    data=people_long,
    x='자치구명',
    y='인구수',
    hue='연령대',
    palette='viridis'
)

# 막대 위 값 표시
for p in ax.patches:
    ax.annotate(
        format(int(p.get_height()), ','),
        (p.get_x() + p.get_width() / 2., p.get_height()),
        ha='center', va='bottom',
        fontsize=10
    )

plt.xlabel('자치구')
plt.ylabel('인구수')
plt.title('자치구별 연령대별 인구수', fontsize=16)
plt.xticks(rotation=0)
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()

st.pyplot(plt)
plt.close()


household = pd.read_csv("house_hold.csv", encoding="utf-8")

# --------------------------
# 2) final_result 기준 자치구 필터링
# --------------------------
filtered_districts = final_result["자치구명"].unique()
household_filtered = household[household["자치구명"].isin(filtered_districts)]

# --------------------------
# 3) Wide -> Long 변환
# --------------------------
household_long = pd.melt(
    household_filtered,
    id_vars=['자치구명'],
    value_vars=['1인가구','2인가구','3인가구','4인가구'],
    var_name='가구유형',
    value_name='가구수'
)

# --------------------------
# 4) 바 그래프 그리기
# --------------------------
st.subheader("자치구별 가구 유형별 가구수")

plt.figure(figsize=(16,6))
ax = sns.barplot(
    data=household_long,
    x='자치구명',
    y='가구수',
    hue='가구유형',
    palette='Set2'
)

# 막대 위 값 표시
for p in ax.patches:
    ax.annotate(
        format(int(p.get_height()), ','),  # 천 단위 콤마
        (p.get_x() + p.get_width() / 2., p.get_height()),
        ha='center', va='bottom',
        fontsize=10
    )

plt.xlabel('자치구')
plt.ylabel('가구수')
plt.title('필터된 매물 자치구별 가구 유형별 가구수', fontsize=16)
plt.xticks(rotation=0)
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()

st.pyplot(plt)
plt.close()


# --------------------------
# 한글 폰트 설정
# --------------------------
plt.rc('font', family='Apple SD Gothic Neo')
plt.rc('axes', unicode_minus=False)   # 마이너스 깨짐 방지

# --------------------------
# final_result 기준 자치구만 선택
# --------------------------
crime = pd.read_csv("final_crime.csv")  # 실제 crime 데이터
crime = crime.drop(columns=["Unnamed: 0"], errors='ignore')

selected_gu = final_result['자치구명'].unique().tolist()
crime_filtered = crime[crime['자치구'].isin(selected_gu)]

crime_long = pd.melt(crime_filtered, id_vars=['자치구'], var_name='연도', value_name='범죄건수')
crime_long['연도'] = crime_long['연도'].astype(int)

# --------------------------
# 막대그래프 그리기
# --------------------------
plt.figure(figsize=(14,7))
ax = sns.barplot(
    data=crime_long,
    x='자치구',
    y='범죄건수',
    ci=None,
    palette='mako'
)

# 막대 위에 값 표시
for p in ax.patches:
    ax.annotate(
        format(int(p.get_height()), ','),
        (p.get_x() + p.get_width() / 2., p.get_height()),
        ha='center', va='bottom',
        fontsize=15
    )

# 제목, 축 레이블, 글자 크기 설정
plt.title("필터된 매물 자치구별 연평균 범죄 건수 (2020~2024)", fontsize=25)
plt.xlabel("자치구", fontsize=0)
plt.ylabel("범죄건수", fontsize=20)
plt.xticks(fontsize=18, rotation=0)  # 기울임 없음
plt.yticks(fontsize=18)
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()

# --------------------------
# Streamlit에 출력
# --------------------------
col4, col5 = st.columns([1, 1])
with col4:
    st.subheader("자치구별 연평균 범죄 건수")
    st.pyplot(plt)
    plt.close()

import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# --------------------------
# 1) 원본 CSV 불러오기
# --------------------------
@st.cache_data
def load_data():
    return pd.read_csv("2018_2024_결측치 제거된 로우데이터.csv", encoding='utf-8')

df = load_data()

# --------------------------
# 2) final_result 기준 자치구만 필터링
# --------------------------
selected_gu = final_result["자치구명"].unique().tolist()
df_filtered = df[df["자치구명"].isin(selected_gu)]

# --------------------------
# 3) 계약연도 · 자치구별 평균 평단가 계산
# --------------------------
avg_result = (
    df_filtered.groupby(["계약 연도", "자치구명"])["평단가(만원)"]
    .mean()
    .reset_index()
    .rename(columns={"평단가(만원)": "평단가평균"})
)

# 연도 정렬
avg_result = avg_result.sort_values(by=["계약 연도", "자치구명"])

# --------------------------
# 4) Seaborn 멀티라인 차트
# --------------------------
plt.figure(figsize=(14, 7))
ax = sns.lineplot(
    data=avg_result,
    x="계약 연도",
    y="평단가평균",
    hue="자치구명",
    marker="o",
    palette="tab10"
)

# --------------------------
# 5) 각 점 위에 값 표시
# --------------------------
for line in ax.get_lines():
    x_data = line.get_xdata()
    y_data = line.get_ydata()
    for x, y in zip(x_data, y_data):
        ax.text(x, y + max(avg_result["평단가평균"])*0.005, f"{y:,.0f}", 
                ha='center', va='bottom', fontsize=15)

plt.title("자치구별 5개년 평균 평단가 변화", fontsize=20)
plt.xlabel("계약 연도", fontsize=17)
plt.ylabel("평단가 평균(만원)", fontsize=17)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()

# --------------------------
# 6) Streamlit 출력
# --------------------------
with col5:
    st.subheader("자치구별 5개년 평균 평단가")
    st.pyplot(plt)
    plt.close()

import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

# --------------------------
# 1) CSV 데이터 불러오기
# --------------------------
@st.cache_data
def load_data():
    return pd.read_csv("2018_2024_결측치 제거된 로우데이터.csv", encoding='utf-8')

df = load_data()

# --------------------------
# 2) 자치구별 연도별 평단가 평균 계산
# --------------------------
price_by_year = (
    df.groupby(["자치구명", "계약 연도"])["평단가(만원)"]
      .mean()
      .reset_index()
      .sort_values(["자치구명", "계약 연도"])
)

# --------------------------
# 3) 2018 → 2024 CAGR 계산
# --------------------------
cagr_list = []

for gu in price_by_year["자치구명"].unique():
    temp = price_by_year[price_by_year["자치구명"] == gu]

    if 2018 in temp["계약 연도"].values and 2024 in temp["계약 연도"].values:
        p0 = temp[temp["계약 연도"] == 2018]["평단가(만원)"].values[0]
        p1 = temp[temp["계약 연도"] == 2024]["평단가(만원)"].values[0]

        if p0 > 0:
            cagr = ((p1 / p0) ** (1 / 6)) - 1  # 6년 기간
            cagr_list.append([gu, p0, p1, cagr * 100])  # % 단위로 변환

cagr_df = pd.DataFrame(cagr_list, columns=["자치구명", "2018평단가", "2024평단가", "CAGR"])

# --------------------------
# 4) Streamlit 레이아웃: 60% / 40% 비율
# --------------------------
col6, col7 = st.columns([3, 2])  # col5 60%, col6 40%

# --------------------------
# 5) 바 차트 (col5)
# --------------------------
with col6:
    st.subheader("자치구별 2018~2024 CAGR(연평균 복합 성장률)")

    plt.figure(figsize=(14, 8))
    ax = sns.barplot(
        data=cagr_df.sort_values("CAGR", ascending=False),
        x="자치구명",
        y="CAGR",
        palette="coolwarm"
    )

    for p in ax.patches:
        ax.annotate(
            f"{p.get_height():.2f}%",
            (p.get_x() + p.get_width() / 2., p.get_height()),
            ha='center',
            va='bottom',
            fontsize=12
        )

    plt.title("서울시 자치구별 평단가 CAGR (2018~2024)", fontsize=20)
    plt.xlabel("자치구")
    plt.ylabel("CAGR (%)")
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()

    st.pyplot(plt)
    plt.close()


with col7:
    st.subheader("자치구별 평단가 CAGR 트리맵")

    fig = px.treemap(
        cagr_df,
        path=["자치구명"],
        values="CAGR",
        color="CAGR",
        color_continuous_scale="RdYlGn_r"
    )

    # 제목 제거, 여백 유지
    fig.update_layout(
        title_text="",  # undefined 대신 빈 문자열
        margin=dict(t=10, l=25, r=25, b=25)  # 필요에 따라 t 조절
    )

    # col6 막대그래프 높이에 맞춤
    st.plotly_chart(fig, use_container_width=True, height=450)
