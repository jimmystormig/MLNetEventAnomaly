namespace MLNetEventAnomaly.Predictors
{
    public sealed class SsaChangePointPredictorData
    {
        public int DayOfWeek;
        public float MinutesSinceMidnight;

        public SsaChangePointPredictorData(int dayOfWeek, float minutesSinceMidnight)
        {
            DayOfWeek = dayOfWeek;
            MinutesSinceMidnight = minutesSinceMidnight;
        }
    }
}