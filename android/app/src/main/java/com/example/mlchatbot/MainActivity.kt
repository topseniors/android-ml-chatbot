package com.example.mlchatbot

import android.os.Bundle
import android.widget.*
import androidx.appcompat.app.AppCompatActivity
import org.json.JSONObject
import org.tensorflow.lite.Interpreter
import java.nio.ByteBuffer
import java.nio.ByteOrder

class MainActivity : AppCompatActivity() {
    private lateinit var interpreter: Interpreter
    private lateinit var metadata: JSONObject
    private lateinit var chatBox: LinearLayout
    private lateinit var input: EditText

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        interpreter = Interpreter(loadModelFile("intent_model.tflite"))
        metadata = JSONObject(assets.open("metadata.json").bufferedReader().use { it.readText() })

        val root = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(24, 24, 24, 24)
        }

        val scroll = ScrollView(this)
        chatBox = LinearLayout(this).apply { orientation = LinearLayout.VERTICAL }
        scroll.addView(chatBox)

        input = EditText(this).apply { hint = "Type message..." }
        val button = Button(this).apply { text = "Send" }

        root.addView(scroll, LinearLayout.LayoutParams(-1, 0, 1f))
        root.addView(input)
        root.addView(button)
        setContentView(root)

        addMessage("Bot", "Hello! Ask me about pricing, refund, or support.")

        button.setOnClickListener {
            val text = input.text.toString().trim()
            if (text.isNotEmpty()) {
                addMessage("You", text)
                addMessage("Bot", getReply(text))
                input.setText("")
            }
        }
    }

    private fun getReply(message: String): String {
        val inputArray = arrayOf(message.lowercase())
        val tags = metadata.getJSONArray("tags")
        val output = Array(1) { FloatArray(tags.length()) }

        interpreter.run(inputArray, output)

        val scores = output[0]
        val bestIndex = scores.indices.maxByOrNull { scores[it] } ?: -1
        val confidence = if (bestIndex >= 0) scores[bestIndex] else 0f
        val threshold = metadata.getDouble("confidence_threshold").toFloat()

        if (bestIndex < 0 || confidence < threshold) {
            return "Sorry, I don't understand. Please ask about pricing, refund, or support."
        }

        val tag = tags.getString(bestIndex)
        val responses = metadata.getJSONObject("responses").getJSONArray(tag)
        return responses.getString(0)
    }

    private fun addMessage(sender: String, text: String) {
        val view = TextView(this).apply {
            this.text = "$sender: $text"
            textSize = 17f
            setPadding(0, 10, 0, 10)
        }
        chatBox.addView(view)
    }

    private fun loadModelFile(fileName: String): ByteBuffer {
        val bytes = assets.open(fileName).readBytes()
        return ByteBuffer.allocateDirect(bytes.size).apply {
            order(ByteOrder.nativeOrder())
            put(bytes)
            rewind()
        }
    }

    override fun onDestroy() {
        interpreter.close()
        super.onDestroy()
    }
}
