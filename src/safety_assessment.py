def assess_safety(accident_risk, weather):
    """
    Combines historical accident risk with current weather
    to generate a current transportation safety assessment.
    """

    # Convert historical accident risk into a numerical score
    accident_scores = {
        "Low": 25,
        "Medium": 55,
        "High": 85
    }

    accident_score = accident_scores.get(accident_risk, 55)

    # Add a weather-risk contribution
    weather_score = 0

    if weather:
        if weather["status"] == "Adverse":
            weather_score = 20
        elif weather["status"] == "Moderate":
            weather_score = 10

    # Combined safety score
    safety_score = min(accident_score + weather_score, 100)

    # Determine final safety level
    if safety_score < 40:
        safety_level = "Low"
    elif safety_score < 70:
        safety_level = "Medium"
    else:
        safety_level = "High"

    # Generate reasons
    reasons = []

    reasons.append(
        f"Historical accident risk is {accident_risk}."
    )

    if weather:
        if weather["status"] == "Adverse":
            reasons.append(
                "Current weather conditions are adverse."
            )
        elif weather["status"] == "Moderate":
            reasons.append(
                "Current weather conditions require caution."
            )
        else:
            reasons.append(
                "Current weather conditions are normal."
            )

        if weather["rain"] > 0:
            reasons.append(
                f"Current rainfall is {weather['rain']:.1f} mm."
            )

        if weather["wind"] >= 25:
            reasons.append(
                f"Current wind speed is {weather['wind']:.1f} km/h."
            )

    # Generate alert message
    if safety_level == "High":
        alert = "HIGH SAFETY RISK: Exercise increased caution while travelling."
    elif safety_level == "Medium":
        alert = "MODERATE SAFETY RISK: Travel with caution."
    else:
        alert = "LOW SAFETY RISK: No elevated safety signal detected."

    return {
        "score": safety_score,
        "level": safety_level,
        "alert": alert,
        "reasons": reasons
    }