namespace MLNetEventAnomaly.Predictors
{
    public class RandomizedPcaPredictorData
    {
        public int DayOfWeek;
        public int Hour;
        public bool Label;

        public RandomizedPcaPredictorData(int dayOfWeek, int hour)
        {
            DayOfWeek = dayOfWeek;
            Hour = hour;
        }
    }
}