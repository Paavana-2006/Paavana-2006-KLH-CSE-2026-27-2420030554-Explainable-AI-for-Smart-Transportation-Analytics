def generate_alert(safety_result):

    if safety_result is None:
        return {
            "type": "info",
            "title": "Safety Status Unavailable",
            "message": "Current safety information could not be generated."
        }

    level = safety_result["level"]

    if level == "High":

        return {
            "type": "high",
            "title": "🚨 HIGH SAFETY ALERT",
            "message": (
                "Elevated transportation safety risk detected. "
                "Exercise increased caution while travelling."
            )
        }

    elif level == "Medium":

        return {
            "type": "medium",
            "title": "⚠️ MODERATE SAFETY ALERT",
            "message": (
                "Moderate transportation risk detected. "
                "Monitor current conditions while travelling."
            )
        }

    else:

        return {
            "type": "low",
            "title": "✅ NORMAL SAFETY STATUS",
            "message": (
                "No elevated transportation safety signal "
                "was detected under the current assessment."
            )
        }