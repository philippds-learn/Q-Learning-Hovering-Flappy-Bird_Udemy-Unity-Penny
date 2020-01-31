using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.Linq;

public class Replay
{
    public List<double> states;
    public double reward;

    public Replay(double distanceTop, double distanceBot, double reward)
    {
        this.states = new List<double>();
        this.states.Add(distanceTop);
        this.states.Add(distanceBot);
        this.reward = reward;
    }
}

public class Brain : MonoBehaviour
{
    public GameObject topBeam;
    public GameObject bottomBeam;

    ANN ann;

    float reward = 0.0f;                                // reward to associate with actions
    List<Replay> replayMemory = new List<Replay>();     // memory - list of past actions and rewards
    int mCapacity = 10000;                              // memory capacity

    float discount = 0.99f;                             // how much future states affect rewards
    float exploreRate = 100.0f;                         // chance of picking random action
    float maxExploreRate = 100.0f;                      // max chance value
    float minExploreRate = 0.01f;                       // min chance value
    float exploreDecay = 0.0001f;                       // chance decay amount for each update

    int failCount = 0;                                  // count when the ball is dropped
    float moveForce = 0.5f;                             // max angle to apply to tilting each update
                                                        // make sure this is large enough so that the q value
                                                        // multiplied by it is enough to recover balance
                                                        // when the ball gets a good speed up

    float timer = 0;                                    // timer to keep track of balancing
    float maxHoverTime = 0;                           // record time ball is kept balanced

    bool crashed = false;
    Vector3 startPos;
    Rigidbody2D rb;

    // Start is called before the first frame update
    void Start()
    {
        this.ann = new ANN(2, 2, 1, 6, 0.2f);
        this.startPos = this.transform.position;
        Time.timeScale = 5.0f;
        this.rb = this.GetComponent<Rigidbody2D>();
    }

    GUIStyle guiStyle = new GUIStyle();
    private void OnGUI()
    {
        this.guiStyle.fontSize = 25;
        this.guiStyle.normal.textColor = Color.white;
        GUI.BeginGroup(new Rect(10, 10, 600, 150));
        GUI.Box(new Rect(0, 0, 140, 140), "Stats", this.guiStyle);
        GUI.Label(new Rect(10, 25, 500, 30), "Fails: " + this.failCount, this.guiStyle);
        GUI.Label(new Rect(10, 50, 500, 30), "Decay Rate: " + this.exploreRate, this.guiStyle);
        GUI.Label(new Rect(10, 75, 500, 30), "Last Best Hover: " + this.maxHoverTime, this.guiStyle);
        GUI.Label(new Rect(10, 100, 500, 30), "This Hover: " + this.timer, this.guiStyle);
        GUI.EndGroup();
    }

    // Update is called once per frame
    void Update()
    {
        if(Input.GetKeyDown("space"))
        {
            ResetBird();
        }
    }

    private void OnCollisionEnter2D(Collision2D collision)
    {
        this.crashed = true;
    }
    private void OnCollisionExit2D(Collision2D collision)
    {
        this.crashed = false;
    }

    void FixedUpdate()
    {
        this.timer += Time.deltaTime;
        List<double> states = new List<double>();
        List<double> qs = new List<double>();

        states.Add(Vector3.Distance(this.transform.position, this.topBeam.transform.position));
        states.Add(Vector3.Distance(this.transform.position, this.bottomBeam.transform.position));

        qs = SoftMax(this.ann.CalcOutput(states));
        double maxQ = qs.Max();
        int maxQIndex = qs.ToList().IndexOf(maxQ);
        this.exploreRate = Mathf.Clamp(this.exploreRate - this.exploreDecay, this.minExploreRate, this.maxExploreRate);

        
        if(Random.Range(0, 100) < this.exploreRate)
        {
            maxQIndex = Random.Range(0, 2);
        }
        

        if(maxQIndex == 0)
        {
            this.rb.AddForce(Vector3.up * this.moveForce * (float)qs[maxQIndex]);
        }
        else if(maxQIndex == 1)
        {
            this.rb.AddForce(Vector3.up * -this.moveForce * (float)qs[maxQIndex]);
        }

        if(this.crashed)
        {
            this.reward = -1.0f;
        }
        else
        {
            this.reward = 0.1f;
        }

        Replay lastMemory = new Replay(Vector3.Distance(this.transform.position, this.topBeam.transform.position),
                                       Vector3.Distance(this.transform.position, this.bottomBeam.transform.position),
                                       this.reward);

        if(this.replayMemory.Count > this.mCapacity)
        {
            this.replayMemory.RemoveAt(0);
        }

        this.replayMemory.Add(lastMemory);

        if(this.crashed)
        {
            for(int i = replayMemory.Count - 1; i >= 0; i--)
            {
                List<double> toutputsOld = new List<double>();
                List<double> toutputsNew = new List<double>();
                toutputsOld = SoftMax(this.ann.CalcOutput(this.replayMemory[i].states));

                double maxQOld = toutputsOld.Max();
                int action = toutputsOld.ToList().IndexOf(maxQOld);

                double feedback;
                if(i == this.replayMemory.Count - 1 || this.replayMemory[i].reward == -1)
                {
                    feedback = this.replayMemory[i].reward;
                }
                else
                {
                    toutputsNew = SoftMax(this.ann.CalcOutput(this.replayMemory[i + 1].states));
                    maxQ = toutputsNew.Max();
                    feedback = (this.replayMemory[i].reward + this.discount * maxQ);
                }

                toutputsOld[action] = feedback;
                this.ann.Train(this.replayMemory[i].states, toutputsOld);
            }

            if(this.timer > this.maxHoverTime)
            {
                this.maxHoverTime = this.timer;
            }

            this.timer = 0;

            this.crashed = false;
            ResetBird();
            this.replayMemory.Clear();
            this.failCount++;
        }
    }

    void ResetBird()
    {
        this.transform.position = this.startPos;
        this.GetComponent<Rigidbody2D>().velocity = new Vector3(0, 0, 0);
    }

    List<double> SoftMax(List<double> values)
    {
        double max = values.Max();

        float scale = 0.0f;

        for(int i = 0; i < values.Count; ++i)
        {
            scale += Mathf.Exp((float)(values[i] - max));
        }

        List<double> result = new List<double>();
        for(int i = 0; i < values.Count; ++i)
        {
            result.Add(Mathf.Exp((float)(values[i] - max)) / scale);
        }

        return result;
    }
}
