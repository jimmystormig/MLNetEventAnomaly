namespace MLNetEventAnomaly.Predictors
{
    public sealed class SsaChangePointPredictorData
    {
        public int DayOfWeek;
        public int Hour;
        public float DoorOpenings;

        public SsaChangePointPredictorData(int dayOfWeek, int hour, int doorOpenings)
        {
            DayOfWeek = dayOfWeek;
            Hour = hour;
            DoorOpenings = doorOpenings;
        }
    }
}