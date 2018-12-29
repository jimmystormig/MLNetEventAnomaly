using System;
using Microsoft.ML.Runtime.Api;
using Microsoft.ML.Transforms;

namespace EventAnomaly
{
    public sealed class Data
    {
        public int DayOfWeek;
        public float MinutesSinceMidnight;

        public Data(int dayOfWeek, float minutesSinceMidnight)
        {
            DayOfWeek = dayOfWeek;
            MinutesSinceMidnight = minutesSinceMidnight;
        }
    }
}