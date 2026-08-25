from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    BaseDocTemplate, Frame, PageBreak, PageTemplate, Paragraph, Spacer,
    Table, TableStyle, Flowable,
)

ROOT = Path(__file__).resolve().parent
TMP = ROOT / "tmp" / "pdfs"
OUT = ROOT / "output" / "pdf"
TMP.mkdir(parents=True, exist_ok=True)
OUT.mkdir(parents=True, exist_ok=True)

NAVY = colors.HexColor("#12355B")
BLUE = colors.HexColor("#1F77B4")
TEAL = colors.HexColor("#16817A")
PALE = colors.HexColor("#EAF2F8")
SLATE = colors.HexColor("#4A5568")
LIGHT = colors.HexColor("#F6F8FA")

data = pd.read_csv(ROOT / "data" / "building_105_meter0.csv")
data["timestamp"] = pd.to_datetime(data["timestamp"])
data["month"] = data["timestamp"].dt.month
data["weekend"] = data["dow"] >= 5

# This mirrors the notebook's defensible decision to remove the dead-flat period
# before weather/model analysis.
model_data = data.loc[~data["timestamp"].between("2016-03-01", "2016-04-20")].copy()
weather = pd.read_csv(ROOT / "data" / "weather_train.csv")
weather = weather.loc[weather["site_id"] == 1, ["timestamp", "air_temperature"]].copy()
weather["timestamp"] = pd.to_datetime(weather["timestamp"])
weather = weather.set_index("timestamp").reindex(
    pd.date_range("2016-01-01", "2016-12-31 23:00", freq="h")
).ffill().rename_axis("timestamp").reset_index()
model_data = model_data.merge(weather, on="timestamp", how="left")

class Chart(Flowable):
    """Compact, vector charts built without a plotting dependency."""
    def __init__(self, kind, width, height):
        super().__init__(); self.kind, self.width, self.height = kind, width, height
    def draw_axes(self, x, y, w, h, title, xlabel, ylabel):
        c = self.canv; c.setStrokeColor(colors.HexColor("#9AA5B1")); c.setLineWidth(.45)
        c.line(x, y, x, y+h); c.line(x, y, x+w, y); c.setFont("Helvetica-Bold", 7.5); c.setFillColor(NAVY)
        c.drawString(x, y+h+10, title); c.setFont("Helvetica", 6.2); c.setFillColor(SLATE)
        c.drawCentredString(x+w/2, y-14, xlabel); c.saveState(); c.translate(x-17, y+h/2); c.rotate(90); c.drawCentredString(0,0,ylabel); c.restoreState()
    def line(self, vals, x, y, w, h, color, ymin=None, ymax=None):
        vals=list(vals); ymin=min(vals) if ymin is None else ymin; ymax=max(vals) if ymax is None else ymax
        rng=max(ymax-ymin, 1e-6); p=self.canv; p.setStrokeColor(color); p.setLineWidth(1.25)
        for i in range(1,len(vals)):
            p.line(x+(i-1)*w/(len(vals)-1), y+(vals[i-1]-ymin)*h/rng, x+i*w/(len(vals)-1), y+(vals[i]-ymin)*h/rng)
    def draw(self):
        c=self.canv; pad=24; w=(self.width-26)/2; h=(self.height-25)/2
        daily=data.groupby("hour")["meter_reading"].mean(); self.draw_axes(pad,self.height/2+7,w,h,"Daily load profile","Hour of day","Average kWh")
        self.line(daily, pad,self.height/2+7,w,h,BLUE,40,110)
        week=data.groupby(["weekend","hour"])["meter_reading"].mean(); x=pad+w+28; self.draw_axes(x,self.height/2+7,w,h,"Weekday versus weekend","Hour of day","Average kWh")
        self.line(week.loc[False],x,self.height/2+7,w,h,BLUE,40,110); self.line(week.loc[True],x,self.height/2+7,w,h,TEAL,40,110)
        c.setFont("Helvetica",5.8); c.setFillColor(BLUE); c.drawString(x+w-55,self.height-4,"Weekday"); c.setFillColor(TEAL); c.drawString(x+w-25,self.height-4,"Weekend")
        y=20; self.draw_axes(pad,y,w,h,"Temperature versus electricity","Air temperature (C)","Meter reading (kWh)")
        sample=model_data.iloc[::20]; c.setFillColor(colors.Color(.12,.47,.71,alpha=.28))
        for _,r in sample.iterrows():
            xx=pad+(r.air_temperature+10)/45*w; yy=y+max(0,min(1,r.meter_reading/180))*h; c.circle(xx,yy,.65,fill=1,stroke=0)
        x=pad+w+28; self.draw_axes(x,y,w,h,"Late-year demand and winter break","Date (Oct-Dec)","Meter reading (kWh)")
        view=data.loc[data.timestamp>=pd.Timestamp("2016-10-01")].iloc[::8]; self.line(view.meter_reading,x,y,w,h,NAVY,0,180)
        c.setFillColor(colors.Color(.96,.64,.38,alpha=.3)); c.rect(x+w*.91,y,w*.09,h,fill=1,stroke=0); c.setFillColor(SLATE); c.setFont("Helvetica",5.6); c.drawString(x+w*.77,y+h-7,"Winter break")

class MetricBars(Flowable):
    def __init__(self, width=6.3*inch, height=2.25*inch): super().__init__(); self.width,self.height=width,height
    def draw(self):
        c=self.canv; x0=42; y0=28; W=self.width-55; H=self.height-58; c.setFont("Helvetica-Bold",9); c.setFillColor(NAVY); c.drawString(x0,y0+H+30,"Test-set error: baseline beats XGBoost")
        c.setStrokeColor(colors.HexColor("#9AA5B1")); c.line(x0,y0,x0,y0+H); c.line(x0,y0,x0+W,y0); c.setFont("Helvetica",6.5); c.setFillColor(SLATE); c.drawString(x0,y0+H+5,"20 kWh")
        values=[(16.8321,10.3927),(17.3108,11.2319)]; labels=["Weekly-lag baseline","XGBoost"]
        for i,(rmse,mae) in enumerate(values):
            cx=x0+W*(.25+.5*i); bw=28
            for j,(val,col,name) in enumerate([(rmse,NAVY,"RMSE"),(mae,TEAL,"MAE")]):
                bx=cx+(j-.5)*bw; bh=val/20*H; c.setFillColor(col); c.rect(bx,y0,bw-4,bh,fill=1,stroke=0); c.setFont("Helvetica",7); c.setFillColor(SLATE); c.drawCentredString(bx+(bw-4)/2,y0+bh+4,f"{val:.1f}")
            c.setFont("Helvetica",7); c.setFillColor(SLATE); c.drawCentredString(cx,y0-12,labels[i])
        c.setFillColor(NAVY); c.rect(x0+W-80,y0+H+18,7,7,fill=1,stroke=0); c.setFillColor(SLATE); c.drawString(x0+W-70,y0+H+19,"RMSE")
        c.setFillColor(TEAL); c.rect(x0+W-40,y0+H+18,7,7,fill=1,stroke=0); c.setFillColor(SLATE); c.drawString(x0+W-30,y0+H+19,"MAE")

styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name="ReportTitle", parent=styles["Title"], fontName="Helvetica-Bold", fontSize=24,
                          leading=28, textColor=NAVY, spaceAfter=5))
styles.add(ParagraphStyle(name="SubTitle", parent=styles["Normal"], fontName="Helvetica", fontSize=10,
                          leading=14, textColor=SLATE, spaceAfter=15))
styles.add(ParagraphStyle(name="H1x", parent=styles["Heading1"], fontName="Helvetica-Bold", fontSize=14,
                          leading=18, textColor=NAVY, spaceBefore=8, spaceAfter=6))
styles.add(ParagraphStyle(name="Bodyx", parent=styles["BodyText"], fontName="Helvetica", fontSize=9.2,
                          leading=13.2, textColor=colors.HexColor("#263238"), spaceAfter=7))
styles.add(ParagraphStyle(name="Small", parent=styles["BodyText"], fontName="Helvetica", fontSize=7.6,
                          leading=9.6, textColor=SLATE))
styles.add(ParagraphStyle(name="TableHeader", parent=styles["BodyText"], fontName="Helvetica-Bold", fontSize=7.6,
                          leading=9.6, textColor=colors.white))
styles.add(ParagraphStyle(name="Callout", parent=styles["BodyText"], fontName="Helvetica-Bold", fontSize=9.3,
                          leading=13, textColor=NAVY, backColor=PALE, borderPadding=8, spaceBefore=4, spaceAfter=9))

def P(text, style="Bodyx"):
    return Paragraph(text, styles[style])

def footer(canvas, doc):
    canvas.saveState()
    canvas.setStrokeColor(colors.HexColor("#CBD5E0")); canvas.line(.6*inch, .52*inch, 7.9*inch, .52*inch)
    canvas.setFont("Helvetica", 7.5); canvas.setFillColor(SLATE)
    canvas.drawString(.6*inch, .33*inch, "ASHRAE Energy Consumption Forecasting | Building 105")
    canvas.drawRightString(7.9*inch, .33*inch, f"Page {doc.page}")
    canvas.restoreState()

doc = BaseDocTemplate(str(OUT / "Aryan_Task1_Energy_Consumption_Writeup.pdf"), pagesize=letter,
                      leftMargin=.6*inch, rightMargin=.6*inch, topMargin=.55*inch, bottomMargin=.7*inch)
frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="main")
doc.addPageTemplates([PageTemplate(id="report", frames=frame, onPage=footer)])

story = []
story += [P("Energy Consumption Forecasting", "ReportTitle"),
          P("ASHRAE dataset | Building 105 electricity demand | Task 1 write-up", "SubTitle"),
          P("1. Building selection and data preparation", "H1x"),
          P("I selected <b>Building 105</b> because it uses electricity meter type 0, has a complete year of <b>8,784 hourly readings</b>, and has no zero readings. It is a <b>five-floor Education</b> building with <b>50,623 sq ft</b> at site 1. The complete series makes it a good choice for a time-based train/test split."),
          P("I kept only electricity data, checked that the building had a full year of readings, and then joined site 1 weather data. I removed a dead-flat period from 1 March to 20 April before training because it did not look like normal building use. This left 7,583 records for the weather and model work."),
          P("2. Exploratory findings", "H1x"),
          Spacer(1, 5),
          Chart("eda", 7.1*inch, 3.8*inch),
          P("Figure 1. Demand rises after 06:00, is highest in the afternoon, and falls overnight. Weekend demand is about 28% lower than weekday demand. Temperature has only a weak link with electricity use (correlation 0.219). Demand also drops sharply during the late-December winter break.", "Small")]
story.append(PageBreak())
story += [P("3. Features and model choice", "H1x"),
          P("The model uses <b>hour of day</b>, <b>day of week</b>, and <b>month</b>, along with the electricity use <b>24 hours ago</b> and <b>168 hours ago</b> (the same hour last week). I used timestamps to create the lag values so the removed March-April period could not create incorrect lag values. After dropping rows without both lags, 7,247 records remained."),
          P("I chose <b>XGBoost regression</b> because electricity use changes in a non-linear way. For example, the effect of 14:00 is different on a weekday than on a weekend. XGBoost can learn these patterns from calendar and lag features without feature scaling. I trained on the first 80% of the data (5,797 rows) and tested on the final 20% (1,450 rows). The rows were never shuffled."),
          P("4. Results and error analysis", "H1x"),
          MetricBars(),
          P("Figure 2. Out-of-sample errors. Lower is better.", "Small"),
          Spacer(1, 7)]
table_data = [[P("Measure", "TableHeader"), P("Weekly-lag baseline", "TableHeader"), P("XGBoost", "TableHeader"), P("Interpretation", "TableHeader")],
              [P("RMSE", "Small"), P("16.83 kWh", "Small"), P("17.31 kWh", "Small"), P("Large misses slightly favor the baseline.", "Small")],
              [P("MAE", "Small"), P("10.39 kWh", "Small"), P("11.23 kWh", "Small"), P("Typical absolute error also favors the baseline.", "Small")]]
t = Table(table_data, colWidths=[.85*inch, 1.38*inch, .9*inch, 3.35*inch])
t.setStyle(TableStyle([("BACKGROUND", (0,0), (-1,0), NAVY), ("TEXTCOLOR", (0,0), (-1,0), colors.white),
                       ("BACKGROUND", (0,1), (-1,-1), LIGHT), ("GRID", (0,0), (-1,-1), .35, colors.HexColor("#CBD5E0")),
                       ("VALIGN", (0,0), (-1,-1), "MIDDLE"), ("LEFTPADDING", (0,0), (-1,-1), 6),
                       ("RIGHTPADDING", (0,0), (-1,-1), 6), ("TOPPADDING", (0,0), (-1,-1), 5), ("BOTTOMPADDING", (0,0), (-1,-1), 5)]))
story += [t, Spacer(1, 10),
          P("The XGBoost model follows normal demand fairly well, but it makes larger errors after 23 December. This is the start of winter break: the building uses less electricity, but the model still expects a normal academic schedule. The 168-hour lag accounts for about 72% of feature importance, so the model mostly repeats the same-hour-last-week baseline instead of improving on it."),
          P("5. Improvements with more time or data", "H1x"),
          P("Next, I would add holiday and academic-calendar flags and predict the <i>change from last week</i> instead of the raw reading. I would also tune XGBoost using time-series validation and test whether temperature helps more when the building is occupied. Building schedules, term dates, HVAC settings, and special-event data would help the model handle winter breaks and other schedule changes."),
          Spacer(1, 8),
          P("Source: analysis reproduced from the accompanying notebook, <i>code/code.ipynb</i>, using Building 105 meter data and ASHRAE site 1 weather data.", "Small")]

doc.build(story)
print(OUT / "Aryan_Task1_Energy_Consumption_Writeup.pdf")
