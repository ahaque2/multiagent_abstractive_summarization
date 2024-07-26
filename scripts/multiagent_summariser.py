from typing import List
from openai import OpenAI
import logging
import os
import time
import json
import copy 

class ComposableAgent:
    """
    This class enables a series of iterative conversations to take place between the user 
    and a persistant LLM agent. 
    """
    def __init__(self, prompt_setup: List[str], max_iterations) -> None:
        """
        Args:
            prompt_setup: A list of prompts used to give the agent their persona and any few shot
            prompt examples etc
        """
        logging.basicConfig(filename="logs/.test",
                    filemode='w',
                    format='%(message)s',
                    level=logging.WARN)
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.WARNING)
        
        self.client = OpenAI(
            api_key=os.environ.get("LAS_API_TOKEN")
        )
        self.model = "gpt-3.5-turbo"
        
        self.messages = [
        ]
        for message in prompt_setup:
            self.messages.append({"role": "user", "content": message})
            response = self.client.chat.completions.create(
                model=self.model,
                messages=self.messages
            )
            self.messages.append({"role": "assistant", "content": response.choices[0].message.content})

        self.logger.debug(json.dumps(self.messages, indent=2))
        self.count = 1
        self.max_iter = max_iterations
        
    def chat(self, message: str) -> str:
        """
        Args:
            message: A prompting string for the LLM
        """
        if self.count<2:
            # Use higher temperature for the first summaries to 
            # encourage diversity of summaries
            temperature = 1.0 #1.2
            print("First summary using t=1.2")
        elif self.count==self.max_iter:
            # On all subsequent iterations use a low temperature to speed
            # up the convergence
            temperature = 0
            print("Final summary using t=0")
        else:
            temperature = 1
            print("Intermediate summary using t=1.0")

        self.messages.append({"role": "user", "content": message})
        while True:
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=self.messages,
                    temperature=temperature
                )
                break
            except:
                # if the OpenAI API throws an error wait befefore retrying 
                time.sleep(0.1)
        self.messages.append({"role": "assistant", "content": response.choices[0].message.content})
        result = response.choices[0].message.content
        self.logger.warning(result)
        self.count += 1
        
        return result
    
def get_multiagent_summary(documents: List[str], personalisation: List[str], num_sentences=5, num_iterations=3) -> str:
    """
    Returns the result of a multiagent summary
    Args:
        documents: An iterable of strings representing the document (or documents to summarise)
        num_sentences: The desired length of the output summary
        num_agents: The number of agents which will collaborate on the summary
        num_iterations: The number of rounds of conversation/debate which the agents will have
    """
    base_prompts = [f"You are a writer who is an expert is creating a short summary of a collection of news articles. The objective of this summary is to capture all the important information from the original articles in {num_sentences} sentences.\n{personal}.\nFormat the summary in a single paragraph.\nDo you understand your role?" for personal in personalisation]
    
    articles = "Article:\n\n".join(documents)
    
    FIRST_PROMPT = f"Summarise this collection of articles:\n{articles}"
    agents = [ComposableAgent([base_prompt], max_iterations=num_iterations) for base_prompt in base_prompts]
    
    iteration_results = []
    for i in range(num_iterations):
        if i==0:
            agent_results = []
            for agent in agents:
                agent_results.append({"content": agent.chat(FIRST_PROMPT)})
            iteration_results.append(agent_results)
        else:
            agent_results = []
            for idx, agent in enumerate(agents):
                summary_list = copy.deepcopy(iteration_results[-1])
                summary_list.pop(idx)
                summary_list = [row["content"] for row in summary_list]
                iter_prompt = "These are the summaries from the other agents:"
                for summary in summary_list:
                    iter_prompt += f"\nAgent response: {summary}"
                iter_prompt += f"\n\nProduce an updated summary which incorporates the best parts of the summaries from the other agents and preserves the best part of your current summary. In evaluating which parts of a summary are good you may consider the contents of the original article. The objective of these summaries is to capture all the important information from the original article in {num_sentences} sentences. Format the summary in a single paragraph. Don't give any explanation, just return the updated summary."
                agent_results.append({"content": agent.chat(iter_prompt)})
            iteration_results.append(agent_results)
    # return the first agents summary from the latest iteration
    return iteration_results[-1][0]["content"]


if __name__ == "__main__":
    docs = [
        """Every incoming visitor’s first instinct upon arriving in New Zealand on Monday morning was to ask: “Where is it?” The fog over large chunks of the country was thick enough to ground many domestic flights, threatening to delay the bleary-eyed UK-based reporters heading down to the South Island for the All Blacks’ first squad announcement. The only thing currently less clear, say the locals, is the immediate outlook for their national rugby side.

        Not for a couple of decades, the greybeards reckon, has there been less certainty around the All Blacks, as they prepare to return to the Test match fray next month. They have not played since last October’s Rugby World Cup final and are missing a Who’s Who of familiar names. Sam Whitelock, Brodie Retallick, Sam Cane, Aaron Smith and Richie Mo’unga are either retired or unavailable. Add in the arrival of a new head coach, Scott Robertson, and something even rarer then New Zealand’s endangered fairy tern hangs in the murky winter air: a discernible sense of Kiwi nervousness.

        It was certainly highly instructive, in that regard, to walk through the drizzle to Christchurch’s Convention Centre – not far away from where Paul O’Connell crept up behind an unsuspecting Alastair Campbell at a team announcement press conference in 2005 and yanked down the former spin doctor’s tracksuit bottoms – and witness the unveiling of Robertson’s determinedly unflashy new captain, Scott Barrett.

        Neither the coaches nor Barrett could have been any friendlier or more welcoming to their two overseas guests but, equally, there was a definite first-day-at-big-school vibe. While Robertson is globally renowned for his break-dancing celebrations after his sides win trophies, the All Black job comes with a whole other layer of pressure. It is only when you find yourself in the full glare of the arc lights, with your inaugural squad being announced live on national television, that the sheer weight of responsibility really kicks in.

        The theory was that the appointment of the charismatic Robertson would make the inevitable post-World Cup rebuilding phase a relatively smooth and, therefore, less stressful period. His record with the Crusaders has been remarkable, with seven consecutive Super Rugby titles between 2017 and 2023. The snag is that the serial winners have fallen away spectacularly since he left last year, rather undermining the cosy notion Robertson could just whistle up a trusty core of Crusaders and be pretty much guaranteed a winning All Black team.

        And talking to a number of those involved in some capacity it was hard not to conclude that England really do have a real window of opportunity in New Zealand. Partly it is a product of the limited preparation time that is restricting Robertson’s options. Partly it is that, in one or two areas, this is an All Blacks squad lacking its old bottomless depth. Above all, though, it is simply that their old aura is slowly being chipped away in a modern world where, as one All Black assistant coach conceded, there is almost nothing new under the tactical sun.

        In addition England haven’t been this well-equipped in New Zealand since 2003, when they came to Wellington and won a memorable pre-World Cup Test with, at one point, just 13 players on the field. Whether it be the Tour of Hell in 1998, knackered bodies and red cards in 2004, off-field shenanigans in 2008, dwarf-tossing and ferry-jumping at the 2011 World Cup or a spot of cabin fever at the end of another endless season in 2014, England have not latterly covered themselves in anything resembling glory.

        New Zealand leave with their silver medal after the Rugby World Cup final against South Africa.
        New Zealand have not played since last October’s World Cup final defeat by South Africa. Photograph: Pavel Golovkin/AP
        Now, suddenly, they are in form, in the mood and in an excellent position to burst a few myths, as Ireland did in 2022. No one, to be clear, is suddenly suggesting New Zealand are a busted flush, merely that once-lofty reputations are now less of a protective shield. Think of Afghanistan beating Australia at the T20 World Cup on Sunday. Or the USA defeating Pakistan. Or Glasgow Warriors going to Loftus Versfeld and beating the Bulls at altitude, as happened in Saturday’s United Rugby Championship final. Base levels of fitness, organisation and skill have risen across modern professional sport and some supposedly bigger fish are being caught unawares.

        Cast the net a little wider and global sporting domination, with increasingly few exceptions, is growing harder to sustain. Opposition video analysts now have endless evidence around which to base their gameplans, as opposed to just crossing their fingers and hoping for the best. As Borthwick made clear before England’s departure from Tokyo, the days of England travelling to New Zealand as rank outsiders are gone.

        Which is what makes this looming two-Test series so fascinating. Borthwick has met Robertson only once, when the pair met for a coffee in London in November 2022, but he already knows what makes these opponents tick. “The Blues won the Super Rugby final in tricky conditions at the weekend and I expect some of the physical confrontational style of the Blues pack to come into the New Zealand team. Then you look at the pace with which the Hurricanes play and the dynamism they have. And the Chiefs are tactically a very smart team. You’d imagine that’ll also be part of it.” Novice Test coaches, scant preparation time. Smart opposition coaches, iffy weather. No wonder so many Kiwis are uneasy about the next few weeks.""",
        """
        A 32-man group has been selected by the new head coach ahead of his first Tests in charge of the national team next month.

        Robertson is looking to rebuild after the loss of several legendary players and he has made some interesting decisions looking towards their July matches.

        Winners

        New captain named

        After much debate, the captaincy has been decided. Scott Barrett follows the greats before him to be named the latest All Blacks skipper, taking over from Sam Cane. Ultimately, in a tight call between the lock and Ardie Savea, it was Barrett’s Crusaders connection which gave him the edge over the back-rower.

        Both are world-class players and inspirational leaders in their own right, so it was otherwise a 50-50 call, but it is not a surprise that Robertson has opted for what he knows. The second-row, despite the odd disciplinary issue, has become one of the best back five forwards in the game and will lead with distinction.

        Uncapped five

        Congratulations to George Bell, Pasilio Tosi, Cortez Ratima, Billy Proctor and Wallace Sititi, who have been selected in an All Blacks squad for the first time after impressive seasons in Super Rugby Pacific.

        Tosi is arguably the main bolter having alternated between a starting and bench role for the Hurricanes, but Robertson evidently sees something special in the tighthead prop, who transitioned from number eight to the front-row a few years ago.

        Elsewhere up front, hooker Bell is rewarded for his superb performances in a struggling Crusaders side, while at the opposite end of the pack – and Super Rugby table – number eight Sititi also gets in. Son of Samoan legend Semo, the 21-year-old is an exciting talent and could well get the nod at the base of the scrum should Ardie Savea be shifted to openside.

        Behind the scrum, Ratima has shone alongside Sititi at the Chiefs, with the duo forming an excellent eight-nine combination, and duly gets a shot in the 32. The half-back probably has the best chance of the uncapped quintet to play a big role next month given New Zealand’s lack of depth in that position.

        Proctor could also come into the reckoning after excelling for the Hurricanes in 2024 but, let’s be honest, Jordie Barrett and Rieko Ioane are the favoured centre pair, with Anton Lienert-Brown a reliable back-up at either 12 or 13.

        Asafo Aumua

        After much promise during his career, 2024 could be the year the explosive hooker finally reaches his potential. Aumua is one of the best age-grade players we’ve ever seen when he produced several barnstorming displays during the 2017 World Rugby U20 Championship, but it has taken time for him to mature in senior rugby.

        With Samisoni Taukei’aho out injured, Dane Coles retired and Codie Taylor entering the latter stages of his Test career, now is the time for Aumua to grasp his opportunity. His work ethic has improved while the nuts and bolts of his game are much better, so let’s hope he can translate that form at Test level.

        TJ Perenara

        After sadly missing the Rugby World Cup through a nasty injury sustained in November 2022, the scrum-half worked remarkably hard to get back fit and he has been rewarded with a deserved recall. As mentioned already, the All Blacks are struggling for depth at scrum-half, but make no mistake, this is very much a form-based decision.

        Since returning to action in March, Perenara has been exceptional for the Hurricanes as they finished at the summit of the Super Rugby table. He remains an elite scrum-half and it wouldn’t be a surprise should he start for New Zealand against England in July.

        Scott Robertson and his trusted Crusaders

        While it was special for the players who were selected, it would have been an equally magical moment for the new All Blacks head coach. After missing out to Ian Foster following the 2019 Rugby World Cup, Robertson finally gets his chance in the hotseat and duly named his first squad on Monday.

        There were a few interesting calls while some were concerned Robertson would lean too heavily on the Crusaders, despite their terrible season. However, it is an ultimately well balanced squad with only a couple of picks potentially being influenced by the All Blacks boss’ background.

        One of those is certainly in the back-row where Ethan Blackadder gets in. The loose forward is an incredible player, so his selection is not unjustified, but other people may have opted against picking the back-rower considering his injury record. However, Robertson quite rightly trusts Blackadder, who played several games at the end of the Super Rugby campaign.

        Sir John Kirwan: ‘What more does Hoskins Sotutu need to do to get an All Blacks call?’

        Losers

        Hoskins Sotutu

        “Disgraceful” and “embarrassing” are already words which have greeted the number eight’s omission on social media from All Blacks supporters, who have been shocked by the news of his absence, and he can quite rightly be annoyed.

        It’s obviously not the Blues star’s skill set with ball in hand or his incredible physical prowess, but there are perhaps still doubts about his work ethic at close quarters. While Sotutu makes big carries and can off-load, you simply need to do more to be a success at Test level, and the nitty-gritty around the contact area is possibly letting him down at the moment.

        Jason Ryan’s influence may also be telling here, too. Sotutu was initially involved after Ryan took charge of the forwards midway through 2022, but he was not in the mix throughout 2023 and will now miss the beginning of the 2024 Test season.

        All Blacks: The telling stats that provide clarity to Hoskins Sotutu’s snub

        Injured players

        Inevitably, a few will miss out through injury but, in this squad update, Sam Cane, Will Jordan, Taukei’aho and Cam Roigard were singled as being on the sidelines, which suggests that they would have been involved had they been fit.

        That is particularly interesting in the case of Cane, who announced that 2024 would be his last Test season after signing a long-term deal in Japan from 2025. For many, the former captain would not be involved at all, but it implies that experienced flanker is still in Robertson’s thoughts.

        As for the other players, it is such a shame to see Jordan continue to struggle with injury, while Roigard was a real breakout star in 2023 and would have been in contention to start at scrum-half this year. Hopefully they will all get their chances in an All Blacks jersey later in the year.

        Ardie Savea

        It has to be said, it’s very much a ‘loser’ with a small ‘l’ given that Savea is still a crucial part of the All Blacks set-up. No doubt the star number eight would have wanted the captaincy, but ultimately it doesn’t lessen his importance to the squad.

        Savea remains the best in the world in his position and has also been handed the role of vice captain to Scott Barrett. However, it would have been a great honour and privilege to have been handed the captaincy responsibility, but this time it wasn’t to be.

        Who will win the New Zealand v England series?

        Hurricanes stars overlooked

        The men from Wellington can perhaps feel the most hard done by as the 50/50 calls generally went against their players. Tosi is the exception, with the tighthead a surprise pick, but the likes of Peter Lakai, Xavier Numia, Brayden Iose, Brett Cameron and Ruben Love were very much putting their hands up. Even Isaia Walker-Leawere is slightly unfortunate considering the All Blacks’ issues at lock.

        The Hurricanes were magnificent in 2024, topping the Super Rugby Pacific following a series of fine displays. Despite missing out on the final with a defeat to the Chiefs in the last-four, their performances probably deserved more recognition. The only caveat is that those aforementioned individuals are young and still have plenty of time to force their way in.

        Shaun Stevenson

        It almost feels inevitable at this point that the Chiefs’ star full-back is going to head abroad and earn a boatload of cash. Foster was clearly never a fan and Robertson has followed suit in not picking the flyer.

        The back three is always very competitive but Stevenson never seems to be rewarded for his fine form. Granted, he has hasn’t been quite as exceptional as last year but, for consistency in performance, it is a shame that he has once again been overlooked.
        """        
    ]
    num_agents = 5
    personalisation=["" for i in range(num_agents)]
    summary = get_multiagent_summary(docs, personalisation)
    print(summary)
